"""
Signal Resolver - Background async task that monitors live prices and resolves signals.
Implements advanced trade management from 2024-2026 crypto perp research:
- Partial TP at 1R with +0.5R trailing stop (avoids BE wick-outs)
- MAE-based early exits (93-95% loss probability detection)
- Structural invalidation triggers
- Dynamic signal expiry by timeframe
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
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
    Background resolver implementing advanced trade management:
    
    1. DYNAMIC EXPIRY: Based on signal timeframe (1m-5m: 4h, 15m-1H: 24h, 4H: 48h, Daily: 7d)
    2. STRUCTURAL INVALIDATION: OB break, FVG full fill, opposite CHoCH
    3. NO BREAKEVEN STOPS: Move to +0.5R after TP1 (avoids wick-outs)
    4. PARTIAL TP: 50% at TP1 (1R), 50% runner with 3x ATR trail
    5. MAE EARLY EXITS: Exit if MAE > 1.0R and PnL < 0, or MAE > 0.7R after 6 bars stall
    """
    
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._price_feeds: Dict[str, Dict[str, float]] = {}  # symbol -> {mark_price, high, low, atr}
        self._candle_data: Dict[str, List[Dict]] = {}  # For structural invalidation checks
        self._websocket_manager: Optional[BybitWebSocketManager] = None
        logger.info("SignalResolver initialized with advanced trade management")
    
    def set_websocket_manager(self, ws_manager: BybitWebSocketManager):
        """Set WebSocket manager for live price feeds."""
        self._websocket_manager = ws_manager
    
    def update_candle_data(self, symbol: str, candle: Dict[str, Any]):
        """Store recent candles for structural invalidation checks."""
        if symbol not in self._candle_data:
            self._candle_data[symbol] = []
        self._candle_data[symbol].append(candle)
        # Keep last 100 candles
        self._candle_data[symbol] = self._candle_data[symbol][-100:]
    
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
    
    def update_price_feed(self, symbol: str, mark_price: float, candle_high: float, candle_low: float, atr_14: Optional[float] = None):
        """Update latest price data for a symbol (called by WebSocket handler)."""
        self._price_feeds[symbol] = {
            "mark_price": mark_price,
            "candle_high": candle_high,
            "candle_low": candle_low,
            "atr_14": atr_14,
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
        active_signals = await tracking_store.get_active_signals()
        
        for tracked_signal in active_signals:
            try:
                price_data = self._price_feeds.get(tracked_signal.symbol)
                
                if not price_data:
                    continue
                
                mark_price = price_data["mark_price"]
                candle_high = price_data["candle_high"]
                candle_low = price_data["candle_low"]
                atr_14 = price_data.get("atr_14")
                
                if tracked_signal.tracking_status == TrackingStatus.PENDING:
                    await self._check_entry_fill(tracked_signal, mark_price)
                    await self._check_expiry(tracked_signal)
                    await self._check_structural_invalidation(tracked_signal, candle_high, candle_low)
                
                elif tracked_signal.tracking_status in [TrackingStatus.ACTIVE, TrackingStatus.HIT_TP1]:
                    # Update MAE/MFE and bars monitoring
                    await tracking_store.update_price_monitoring(
                        tracked_signal.signal_id,
                        mark_price,
                        candle_high,
                        candle_low,
                    )
                    
                    # Check MAE early exits FIRST (before TP/SL checks)
                    await self._check_mae_early_exit(tracked_signal, mark_price)
                    
                    # Check TP1 hit and apply partial logic
                    if tracked_signal.tracking_status == TrackingStatus.ACTIVE:
                        await self._check_tp1_hit_partial(tracked_signal, mark_price, candle_high, candle_low, atr_14)
                    
                    # Check TP2 hit (hard target)
                    await self._check_tp2_hard(tracked_signal, mark_price)
                    
                    # Check trailing stop hit (for runner after TP1)
                    await self._check_trailing_stop_hit(tracked_signal, mark_price, candle_high, candle_low, atr_14)
                    
                    # Check SL hit
                    await self._check_sl_hit(tracked_signal, mark_price)
                    
                    # Check time exit (48 hours timeout)
                    await self._check_time_exit(tracked_signal, mark_price)
                    
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
    
    async def _check_expiry(self, signal: TrackedSignal):
        """Check if signal has expired based on dynamic expiry by timeframe."""
        now = datetime.now(timezone.utc)
        
        if now > signal.expiry_time:
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.EXPIRED,
                exit_reason="EXPIRED",
            )
            signal_output_manager.active_signals.pop(signal.signal_id, None)
            logger.info(f"Signal {signal.signal_id[:8]} EXPIRED (time limit)")
    
    async def _check_structural_invalidation(
        self, 
        signal: TrackedSignal, 
        candle_high: float, 
        candle_low: float,
    ):
        """
        Check for structural invalidation triggers:
        1. OB break: Price breaks and closes beyond OB low (longs) or high (shorts)
        2. FVG full fill: Price trades through entire FVG gap
        3. Opposite CHoCH: 15m Change of Character against signal direction
        
        For MVP, we use price threshold heuristics until full market structure engine is piped in.
        """
        risk_r = abs(signal.entry_price - signal.stop_loss)
        invalidation_reason = None
        
        if signal.direction == "LONG":
            # OB invalidation: close below OB midpoint (approximated as entry - 0.5R)
            ob_invalidation_level = signal.entry_price - (risk_r * 0.5)
            if candle_low < ob_invalidation_level:
                invalidation_reason = "OB_BREAK"
            
            # FVG invalidation: full fill of FVG (price trades through entry to stop)
            fvg_invalidated = candle_low <= signal.stop_loss
            if fvg_invalidated and not invalidation_reason:
                invalidation_reason = "FVG_FULL_FILL"
        
        else:  # SHORT
            ob_invalidation_level = signal.entry_price + (risk_r * 0.5)
            if candle_high > ob_invalidation_level:
                invalidation_reason = "OB_BREAK"
            
            fvg_invalidated = candle_high >= signal.stop_loss
            if fvg_invalidated and not invalidation_reason:
                invalidation_reason = "FVG_FULL_FILL"
        
        # TODO: Add CHoCH detection when structure module is integrated
        # For now, check if 15m candle closed against signal by > 0.7R
        
        if invalidation_reason:
            exit_price = candle_low if signal.direction == "LONG" else candle_high
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.INVALIDATED,
                exit_price=exit_price,
                exit_reason=f"INVALIDATED_{invalidation_reason}",
            )
            signal.invalidation_reason = invalidation_reason
            logger.warning(f"Signal {signal.signal_id[:8]} INVALIDATED: {invalidation_reason}")
    
    async def _check_mae_early_exit(self, signal: TrackedSignal, mark_price: float):
        """
        MAE-based early exit rules from research:
        1. HARD EXIT: MAE > 1.0R AND current PnL < 0 (93-95% loss probability)
        2. STALL EXIT: MAE > 0.7R AND no +0.5R profit after 6 bars
        """
        if not signal.mae_r or not signal.entry_filled_at:
            return
        
        current_pnl_r = signal.calculate_pnl_r(mark_price)
        
        # Rule 1: Hard MAE exit
        if signal.mae_r > settings.MAE_HARD_EXIT_R and current_pnl_r < 0:
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.EARLY_EXIT_MAE,
                exit_price=mark_price,
                exit_reason=f"MAE_HARD_EXIT (MAE={signal.mae_r:.2f}R)",
            )
            logger.warning(
                f"Signal {signal.signal_id[:8]} EARLY_EXIT_MAE: "
                f"MAE={signal.mae_r:.2f}R, PnL={current_pnl_r:.2f}R"
            )
            return
        
        # Rule 2: Stall exit
        if (signal.mae_r > settings.MAE_STALL_EXIT_R and 
            signal.bars_since_entry > settings.MAX_BARS_WITHOUT_PROFIT and
            current_pnl_r < 0.5):
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.EARLY_EXIT_STALL,
                exit_price=mark_price,
                exit_reason=f"STALL_EXIT (MAE={signal.mae_r:.2f}R, bars={signal.bars_since_entry})",
            )
            logger.warning(
                f"Signal {signal.signal_id[:8]} EARLY_EXIT_STALL: "
                f"MAE={signal.mae_r:.2f}R, bars={signal.bars_since_entry}, PnL={current_pnl_r:.2f}R"
            )
    
    async def _check_tp1_hit_partial(
        self, 
        signal: TrackedSignal, 
        mark_price: float,
        candle_high: float,
        candle_low: float,
        atr_14: Optional[float],
    ):
        """
        Partial TP logic at 1R:
        1. Mark 50% position as closed at TP1
        2. Record partial_pnl_r = 1.0R * 0.5 = 0.5R contribution
        3. Move stop to +0.5R (NOT breakeven) to avoid wick-outs
        4. Set up trailing stop for remaining 50% runner
        """
        tp1 = signal.take_profit_1
        hit_tp1 = False
        
        if signal.direction == "LONG" and candle_high >= tp1:
            hit_tp1 = True
        elif signal.direction == "SHORT" and candle_low <= tp1:
            hit_tp1 = True
        
        if not hit_tp1:
            return
        
        # Record TP1 hit time
        now = datetime.now(timezone.utc)
        if signal.entry_filled_at:
            signal.time_to_tp1_seconds = int((now - signal.entry_filled_at).total_seconds())
        signal.tp1_hit_at = now
        
        # Calculate partial PnL (50% of position at 1R)
        partial_pnl_r = 1.0 * 0.5  # 50% of position gained 1R
        signal.partial_pnl_r = partial_pnl_r
        signal.position_remaining_pct = 0.5
        
        if signal.trail_after_tp1:
            # Move stop to +0.5R (NOT exact breakeven)
            new_stop = signal.calculate_be_offset_stop()
            signal.trailing_stop_price = new_stop
            logger.info(
                f"Signal {signal.signal_id[:8]} HIT_TP1: Partial PnL={partial_pnl_r:.2f}R, "
                f"stop moved to +0.5R @ {new_stop:.2f}"
            )
        else:
            logger.info(f"Signal {signal.signal_id[:8]} HIT_TP1: Partial PnL={partial_pnl_r:.2f}R")
        
        # Update status
        await tracking_store.save_signal(signal)
    
    async def _check_tp2_hard(self, signal: TrackedSignal, mark_price: float):
        """Check if hard TP2 target was hit (full win on remaining position)."""
        if signal.tracking_status != TrackingStatus.HIT_TP1:
            return
        
        tp2 = signal.take_profit_2
        hit_tp2 = False
        
        if signal.direction == "LONG" and mark_price >= tp2:
            hit_tp2 = True
        elif signal.direction == "SHORT" and mark_price <= tp2:
            hit_tp2 = True
        
        if hit_tp2:
            # Calculate total PnL: partial (0.5R) + runner (remaining 50% at ~2R)
            runner_pnl_r = signal.calculate_pnl_r(tp2) * 0.5  # 50% of position
            total_pnl_r = signal.partial_pnl_r + runner_pnl_r
            
            signal.runner_pnl_r = runner_pnl_r
            signal.total_pnl_r = total_pnl_r
            
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.WIN_TP2_HARD,
                exit_price=tp2,
                exit_reason="TAKE_PROFIT_TP2",
            )
            logger.info(f"Signal {signal.signal_id[:8]} WIN_TP2_HARD: Total={total_pnl_r:.2f}R")
    
    async def _check_trailing_stop_hit(
        self,
        signal: TrackedSignal,
        mark_price: float,
        candle_high: float,
        candle_low: float,
        atr_14: Optional[float],
    ):
        """Check if trailing stop was hit on the runner position (after TP1)."""
        if signal.tracking_status != TrackingStatus.HIT_TP1:
            return
        
        if not signal.trailing_stop_price:
            # Calculate initial trailing stop if not set
            if signal.trail_after_tp1:
                signal.trailing_stop_price = signal.calculate_trailing_stop(
                    atr_14=atr_14,
                    atr_multiplier=settings.TRAIL_ATR_MULTIPLIER
                )
            return
        
        # Update trailing stop based on price movement
        new_trail = signal.calculate_trailing_stop(
            atr_14=atr_14,
            atr_multiplier=settings.TRAIL_ATR_MULTIPLIER
        )
        
        # Trail only moves in favorable direction
        if signal.direction == "LONG":
            signal.trailing_stop_price = max(signal.trailing_stop_price, new_trail)
            hit_trail = candle_low <= signal.trailing_stop_price
        else:
            signal.trailing_stop_price = min(signal.trailing_stop_price, new_trail)
            hit_trail = candle_high >= signal.trailing_stop_price
        
        if hit_trail:
            # Calculate runner PnL at trailing stop exit
            runner_pnl_r = signal.calculate_pnl_r(signal.trailing_stop_price) * 0.5
            total_pnl_r = (signal.partial_pnl_r or 0) + runner_pnl_r
            
            signal.runner_pnl_r = runner_pnl_r
            signal.total_pnl_r = total_pnl_r
            
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.WIN_TP2_TRAILED,
                exit_price=signal.trailing_stop_price,
                exit_reason=f"TRAILING_STOP ({settings.TRAIL_ATR_MULTIPLIER}xATR)",
            )
            logger.info(
                f"Signal {signal.signal_id[:8]} WIN_TP2_TRAILED: "
                f"Total={total_pnl_r:.2f}R @ {signal.trailing_stop_price:.2f}"
            )
    
    async def _check_sl_hit(self, signal: TrackedSignal, mark_price: float):
        """Check if stop loss was hit."""
        # Use current stop (may have been trailed)
        current_stop = signal.get_current_stop_loss()
        
        hit_sl = False
        if signal.direction == "LONG" and mark_price <= current_stop:
            hit_sl = True
        elif signal.direction == "SHORT" and mark_price >= current_stop:
            hit_sl = True
        
        if hit_sl:
            pnl_r = signal.calculate_pnl_r(current_stop)
            
            # If after TP1, calculate partial + runner loss
            if signal.tracking_status == TrackingStatus.HIT_TP1:
                runner_pnl_r = pnl_r * 0.5
                total_pnl_r = (signal.partial_pnl_r or 0) + runner_pnl_r
                signal.runner_pnl_r = runner_pnl_r
                signal.total_pnl_r = total_pnl_r
            else:
                signal.total_pnl_r = pnl_r
            
            await tracking_store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.LOSS_SL,
                exit_price=current_stop,
                exit_reason="STOP_LOSS",
            )
            logger.warning(f"Signal {signal.signal_id[:8]} LOSS @ SL: {pnl_r:.2f}R")
    
    async def _check_time_exit(self, signal: TrackedSignal, mark_price: float):
        """Check if time exit condition is met (48 hours timeout)."""
        if not signal.entry_filled_at:
            return
        
        now = datetime.now(timezone.utc)
        hours_since_entry = (now - signal.entry_filled_at).total_seconds() / 3600
        
        # Time exit: close if PnL < +0.5R after 48 hours
        if hours_since_entry >= 48:
            pnl_r = signal.calculate_pnl_r(mark_price)
            
            if pnl_r < 0.5:
                # Calculate final PnL considering partial if applicable
                if signal.tracking_status == TrackingStatus.HIT_TP1:
                    runner_pnl_r = pnl_r * 0.5
                    total_pnl_r = (signal.partial_pnl_r or 0) + runner_pnl_r
                    signal.runner_pnl_r = runner_pnl_r
                    signal.total_pnl_r = total_pnl_r
                else:
                    signal.total_pnl_r = pnl_r
                
                await tracking_store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.TIME_EXIT,
                    exit_price=mark_price,
                    exit_reason="TIME_EXIT_48H",
                )
                logger.info(
                    f"Signal {signal.signal_id[:8]} TIME EXIT: {hours_since_entry:.1f}h, "
                    f"PnL={pnl_r:.2f}R"
                )


# Singleton instance
signal_resolver = SignalResolver()
