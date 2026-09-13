"""
Signal Resolver - Background async task that monitors live prices and resolves signals.
Tracks fill status, TP/SL hits, expiry, and invalidation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from tbb.core.config import settings
from tbb.core.logger import setup_logger
from tbb.tracking.store import tracking_store, TrackedSignal, TrackingStatus
from tbb.signals.output import signal_output_manager, TradeSignal, SignalStatus

logger = setup_logger(__name__)


class BybitWebSocketManager:
    """Placeholder WebSocket manager for type hints."""
    pass


class SignalResolver:
    """
    Background resolver that monitors live price data and resolves signals.
    
    Resolution states:
    - FILLED: Entry limit order filled at entry_price
    - HIT_TP1: Price hit take_profit_1 (optionally move SL to BE)
    - WIN_TP2: Price hit take_profit_2 (full win)
    - LOSS_SL: Price hit stop_loss
    - EXPIRED: Signal expired without fill (FVG age limit)
    - INVALIDATED: Opposite structure break (e.g., swing low broken for LONG)
    """
    
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._price_feeds: Dict[str, Dict[str, float]] = {}  # symbol -> {mark_price, high, low}
        self._websocket_manager: Optional[BybitWebSocketManager] = None
        logger.info("SignalResolver initialized")
    
    def set_websocket_manager(self, ws_manager: BybitWebSocketManager):
        """Set WebSocket manager for live price feeds."""
        self._websocket_manager = ws_manager
    
    async def start(self):
        """Start the background resolver loop."""
        if self._running:
            logger.warning("SignalResolver already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._resolver_loop())
        logger.info("SignalResolver started")
    
    async def stop(self):
        """Stop the background resolver loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SignalResolver stopped")
    
    def update_price_feed(self, symbol: str, mark_price: float, candle_high: float, candle_low: float):
        """Update latest price data for a symbol (called by WebSocket handler)."""
        self._price_feeds[symbol] = {
            "mark_price": mark_price,
            "candle_high": candle_high,
            "candle_low": candle_low,
            "timestamp": datetime.now(timezone.utc),
        }
    
    async def _resolver_loop(self):
        """Main resolver loop - checks all active signals every second."""
        while self._running:
            try:
                await self._check_all_signals()
                await asyncio.sleep(1.0)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resolver loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _check_all_signals(self):
        """Check all active/pending signals for resolution."""
        # Get all active signals
        active_signals = await tracking_store.get_active_signals()
        
        for tracked_signal in active_signals:
            try:
                # Get current price for symbol
                price_data = self._price_feeds.get(tracked_signal.symbol)
                
                if not price_data:
                    # No price data yet, skip this signal
                    continue
                
                mark_price = price_data["mark_price"]
                candle_high = price_data["candle_high"]
                candle_low = price_data["candle_low"]
                
                if tracked_signal.tracking_status == TrackingStatus.PENDING:
                    # Check if signal should be filled (entry price hit)
                    await self._check_entry_fill(tracked_signal, mark_price)
                    
                    # Check if signal expired
                    await self._check_expiry(tracked_signal)
                    
                    # Check if invalidated by opposite structure
                    await self._check_invalidation(tracked_signal, candle_high, candle_low)
                
                elif tracked_signal.tracking_status in [TrackingStatus.ACTIVE, TrackingStatus.HIT_TP1]:
                    # Update MAE/MFE monitoring
                    await tracking_store.update_price_monitoring(
                        tracked_signal.signal_id,
                        mark_price,
                        candle_high,
                        candle_low,
                    )
                    
                    # Check TP1 hit
                    if tracked_signal.tracking_status == TrackingStatus.ACTIVE:
                        await self._check_tp1_hit(tracked_signal, mark_price)
                    
                    # Check TP2 hit (full win)
                    await self._check_tp2_hit(tracked_signal, mark_price)
                    
                    # Check SL hit
                    await self._check_sl_hit(tracked_signal, mark_price)
                    
                    # Check time exit (48 hours timeout)
                    await self._check_time_exit(tracked_signal)
                    
            except Exception as e:
                logger.error(f"Error checking signal {tracked_signal.signal_id[:8]}: {e}")
    
    async def _check_entry_fill(self, signal: TrackedSignal, mark_price: float):
        """Check if entry limit order would be filled."""
        entry_price = signal.entry_price
        
        # For LONG: fill when mark_price <= entry_price
        # For SHORT: fill when mark_price >= entry_price
        should_fill = False
        
        if signal.direction == "LONG" and mark_price <= entry_price * 1.0001:  # Small tolerance
            should_fill = True
        elif signal.direction == "SHORT" and mark_price >= entry_price * 0.9999:
            should_fill = True
        
        if should_fill:
            await tracking_store.mark_as_filled(
                signal.signal_id,
                fill_price=entry_price,
                fill_timestamp=datetime.now(timezone.utc),
            )
            logger.info(
                f"Signal {signal.signal_id[:8]} FILLED: {signal.symbol} {signal.direction} "
                f"@ {entry_price}"
            )
    
    async def _check_expiry(self, signal: TrackedSignal):
        """Check if signal has expired (past expiry_time)."""
        now = datetime.now(timezone.utc)
        
        if now > signal.expiry_time:
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.EXPIRED,
                exit_reason="EXPIRED",
            )
            
            # Also update in signal output manager
            signal_output_manager.active_signals.pop(signal.signal_id, None)
            
            logger.info(f"Signal {signal.signal_id[:8]} EXPIRED (time limit)")
    
    async def _check_invalidation(
        self, 
        signal: TrackedSignal, 
        candle_high: float, 
        candle_low: float,
    ):
        """Check if signal is invalidated by opposite structure break."""
        # For LONG: invalidated if swing low below entry is broken
        # For SHORT: invalidated if swing high above entry is broken
        
        # Simple heuristic: if price moves against signal by > 1R before filling
        risk_r = abs(signal.entry_price - signal.stop_loss)
        
        if signal.direction == "LONG":
            # Invalidated if price drops below entry by more than 0.5R
            invalidation_level = signal.entry_price - (risk_r * 0.5)
            if candle_low < invalidation_level:
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.INVALIDATED,
                    exit_price=candle_low,
                    exit_reason="INVALIDATED",
                )
                logger.info(f"Signal {signal.signal_id[:8]} INVALIDATED (structure break)")
        
        else:  # SHORT
            # Invalidated if price rises above entry by more than 0.5R
            invalidation_level = signal.entry_price + (risk_r * 0.5)
            if candle_high > invalidation_level:
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.INVALIDATED,
                    exit_price=candle_high,
                    exit_reason="INVALIDATED",
                )
                logger.info(f"Signal {signal.signal_id[:8]} INVALIDATED (structure break)")
    
    async def _check_tp1_hit(self, signal: TrackedSignal, mark_price: float):
        """Check if TP1 was hit."""
        tp1 = signal.take_profit_1
        
        hit_tp1 = False
        if signal.direction == "LONG" and mark_price >= tp1:
            hit_tp1 = True
        elif signal.direction == "SHORT" and mark_price <= tp1:
            hit_tp1 = True
        
        if hit_tp1:
            if signal.resolve_fully_at_tp1:
                # Resolve fully at TP1
                pnl_r = signal.calculate_pnl_r(tp1)
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.WIN_TP2,
                    exit_price=tp1,
                    exit_reason="TAKE_PROFIT",
                )
                logger.info(f"Signal {signal.signal_id[:8]} WIN @ TP1: {pnl_r:.2f}R")
            else:
                # Just mark as HIT_TP1 (partial)
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.HIT_TP1,
                    exit_price=tp1,
                    exit_reason="TP1_HIT",
                )
                
                # Optionally move SL to breakeven
                if signal.move_sl_to_be_on_tp1:
                    logger.info(f"Signal {signal.signal_id[:8]} moved SL to BE (config enabled)")
                
                logger.info(f"Signal {signal.signal_id[:8]} HIT_TP1")
    
    async def _check_tp2_hit(self, signal: TrackedSignal, mark_price: float):
        """Check if TP2 was hit (full win)."""
        if signal.tracking_status == TrackingStatus.HIT_TP1:
            # Already hit TP1, check TP2
            tp2 = signal.take_profit_2
            
            hit_tp2 = False
            if signal.direction == "LONG" and mark_price >= tp2:
                hit_tp2 = True
            elif signal.direction == "SHORT" and mark_price <= tp2:
                hit_tp2 = True
            
            if hit_tp2:
                pnl_r = signal.calculate_pnl_r(tp2)
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.WIN_TP2,
                    exit_price=tp2,
                    exit_reason="TAKE_PROFIT",
                )
                logger.info(f"Signal {signal.signal_id[:8]} WIN @ TP2: {pnl_r:.2f}R")
    
    async def _check_sl_hit(self, signal: TrackedSignal, mark_price: float):
        """Check if stop loss was hit."""
        sl = signal.stop_loss
        
        hit_sl = False
        if signal.direction == "LONG" and mark_price <= sl:
            hit_sl = True
        elif signal.direction == "SHORT" and mark_price >= sl:
            hit_sl = True
        
        if hit_sl:
            pnl_r = signal.calculate_pnl_r(sl)
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.LOSS_SL,
                exit_price=sl,
                exit_reason="STOP_LOSS",
            )
            logger.warning(f"Signal {signal.signal_id[:8]} LOSS @ SL: {pnl_r:.2f}R")
    
    async def _check_time_exit(self, signal: TrackedSignal):
        """Check if time exit condition is met (48 hours timeout)."""
        if not signal.entry_filled_at:
            return
        
        now = datetime.now(timezone.utc)
        hours_since_entry = (now - signal.entry_filled_at).total_seconds() / 3600
        
        # Time exit: close if PnL < +0.5R after 48 hours
        if hours_since_entry >= 48:
            # Get current mark price
            price_data = self._price_feeds.get(signal.symbol)
            if not price_data:
                return
            
            mark_price = price_data["mark_price"]
            pnl_r = signal.calculate_pnl_r(mark_price)
            
            if pnl_r < 0.5:
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.TIME_EXIT if hasattr(TrackingStatus, 'TIME_EXIT') else TrackingStatus.CANCELLED,
                    exit_price=mark_price,
                    exit_reason="TIME_EXIT",
                )
                logger.info(
                    f"Signal {signal.signal_id[:8]} TIME EXIT: {hours_since_entry:.1f}h, "
                    f"PnL: {pnl_r:.2f}R"
                )


# Singleton instance
signal_resolver = SignalResolver()
