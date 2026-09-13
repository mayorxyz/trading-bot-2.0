"""
Order Lifecycle Manager - Orchestrates sync between local SignalRecord DB and exchange orders.
Maps paper tracker states to exchange actions (place, modify, cancel orders).
Implements advanced trade management: +0.5R rule, partial TPs, MAE early exits.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum
import uuid

from tbb.core.config import settings
from tbb.core.logger import setup_logger
from tbb.tracking.store import SignalTrackingStore, TrackedSignal, TrackingStatus
from tbb.execution.exchange_client import (
    get_exchange_client,
    ExchangeClient,
    OrderSide,
    OrderType,
)
from tbb.execution.position_sizer import calculate_position_size, validate_position_against_limits

logger = setup_logger(__name__)


class LifecycleState(str, Enum):
    """State machine for order lifecycle."""
    INITIALIZED = "initialized"  # Signal received, waiting to place entry
    ENTRY_ORDER_PLACED = "entry_order_placed"  # Limit order on book
    ENTRY_FILLED = "entry_filled"  # Position active
    TP1_HIT = "tp1_hit"  # First take profit hit, waiting to adjust stop
    TP2_ACTIVE = "tp2_active"  # Runner position with trailing stop
    COMPLETED = "completed"  # Fully resolved
    CANCELLED = "cancelled"  # Cancelled before fill
    INVALIDATED = "invalidated"  # Structural invalidation
    EMERGENCY_EXIT = "emergency_exit"  # MAE/stall early exit


class OrderLifecycleManager:
    """
    Manages the full lifecycle of a signal from generation to resolution.
    Syncs local DB state with exchange orders.
    
    Lifecycle:
    1. NEW signal → Place LIMIT entry order
    2. If EXPIRED/INVALIDATED before fill → CANCEL entry order
    3. On ENTRY FILL → Place initial SL and TP1 (50% position)
    4. On TP1 HIT → Move SL to +0.5R, place TP2/trailing for remaining 50%
    5. On TP2/TRAIL HIT → Mark COMPLETED
    6. On SL HIT → Mark LOSS_SL
    7. On MAE/STALL trigger → MARKET close entire position
    """
    
    def __init__(self):
        self.store = SignalTrackingStore()
        self.exchange = get_exchange_client()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._signal_states: Dict[str, LifecycleState] = {}
        
        logger.info("OrderLifecycleManager initialized")
    
    async def start(self):
        """Start the background lifecycle monitoring loop."""
        if self._running:
            logger.warning("OrderLifecycleManager already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._lifecycle_loop())
        logger.info("OrderLifecycleManager started")
    
    async def stop(self):
        """Stop the lifecycle monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OrderLifecycleManager stopped")
    
    async def _lifecycle_loop(self):
        """Background loop to monitor signals and sync with exchange."""
        while self._running:
            try:
                # Check kill switch
                if settings.KILL_SWITCH:
                    logger.critical("Kill switch activated - executing emergency shutdown")
                    await self.exchange.emergency_kill_switch()
                    self._running = False
                    break
                
                # Get all active/pending signals
                active_signals = await self.store.get_active_signals()
                
                for signal in active_signals:
                    try:
                        await self._process_signal(signal)
                    except Exception as e:
                        logger.error(f"Error processing signal {signal.signal_id}: {e}")
                
                # Sleep between cycles
                await asyncio.sleep(2.0)  # Check every 2 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lifecycle loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_signal(self, signal: TrackedSignal):
        """Process a single signal through its lifecycle."""
        signal_id = signal.signal_id
        current_state = self._signal_states.get(signal_id, LifecycleState.INITIALIZED)
        
        # Check for expiry or invalidation
        now = datetime.now(timezone.utc)
        if signal.tracking_status == TrackingStatus.PENDING:
            if now > signal.expiry_time:
                logger.info(f"Signal {signal_id[:8]} expired - cancelling entry order")
                await self._cancel_entry_order(signal)
                await self.store.update_tracking_status(
                    signal_id, TrackingStatus.EXPIRED, exit_reason="TIME_EXPIRY"
                )
                self._signal_states[signal_id] = LifecycleState.CANCELLED
                return
            
            # Check structural invalidation (if resolver marked it)
            if signal.invalidation_reason:
                logger.info(f"Signal {signal_id[:8]} invalidated: {signal.invalidation_reason}")
                await self._cancel_entry_order(signal)
                await self.store.update_tracking_status(
                    signal_id, TrackingStatus.INVALIDATED, exit_reason=signal.invalidation_reason
                )
                self._signal_states[signal_id] = LifecycleState.INVALIDATED
                return
        
        # State machine transitions
        if current_state == LifecycleState.INITIALIZED and signal.tracking_status == TrackingStatus.PENDING:
            # Place entry order if AUTO_EXECUTE is enabled
            if settings.AUTO_EXECUTE:
                await self._place_entry_order(signal)
            else:
                logger.debug(f"Signal {signal_id[:8]} awaiting fill (AUTO_EXECUTE={settings.AUTO_EXECUTE})")
        
        elif current_state == LifecycleState.ENTRY_ORDER_PLACED:
            # Monitor for fill
            await self._check_entry_fill(signal)
        
        elif current_state == LifecycleState.ENTRY_FILLED:
            # Monitor for TP1 hit, SL hit, or MAE exit
            await self._monitor_active_position(signal)
        
        elif current_state == LifecycleState.TP1_HIT:
            # TP1 hit, managing runner position
            await self._monitor_runner_position(signal)
        
        elif current_state == LifecycleState.TP2_ACTIVE:
            # Monitor trailing stop
            await self._check_trail_exit(signal)
    
    async def _place_entry_order(self, signal: TrackedSignal):
        """Place LIMIT entry order for a new signal."""
        if not settings.AUTO_EXECUTE:
            return
        
        # Check portfolio limits
        open_positions = await self.exchange.get_all_positions()
        can_open, reason = validate_position_against_limits(
            signal.symbol,
            signal.risk_reward * signal.entry_price,  # Rough estimate
            open_positions,
        )
        
        if not can_open:
            logger.warning(f"Cannot place entry for {signal.signal_id[:8]}: {reason}")
            return
        
        # Calculate position size
        account_balance = await self.exchange.get_account_balance()
        equity = account_balance.get("equity", 1000.0)  # Default $1k for testing
        
        try:
            size_result = calculate_position_size(
                account_equity=equity,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                side="LONG" if signal.direction == "LONG" else "SHORT",
                symbol=signal.symbol,
                leverage=settings.DEFAULT_LEVERAGE,
                risk_per_trade_pct=settings.RISK_PER_TRADE_PCT,
            )
        except ValueError as e:
            logger.error(f"Position size calculation failed: {e}")
            return
        
        # Place limit order
        side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL
        
        try:
            order_response = await self.exchange.place_limit_order(
                symbol=signal.symbol,
                side=side,
                price=signal.entry_price,
                qty=size_result.quantity,
                reduce_only=False,
                time_in_force="GTC",
            )
            
            order_id = order_response.get("orderId")
            
            # Store exchange order ID in signal record
            signal.exchange_entry_order_id = order_id
            await self.store.save_signal(signal)
            
            self._signal_states[signal.signal_id] = LifecycleState.ENTRY_ORDER_PLACED
            logger.info(
                f"Entry order placed for {signal.signal_id[:8]}: "
                f"{signal.symbol} {side.value} {size_result.quantity}@{signal.entry_price} | OrderID: {order_id}"
            )
        
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
    
    async def _cancel_entry_order(self, signal: TrackedSignal):
        """Cancel entry order when signal expires or is invalidated."""
        if not signal.exchange_entry_order_id:
            logger.debug(f"No entry order ID stored for {signal.signal_id[:8]}")
            return
        
        try:
            await self.exchange.cancel_order(signal.symbol, signal.exchange_entry_order_id)
            logger.info(f"Entry order cancelled for {signal.signal_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to cancel entry order: {e}")
    
    async def _check_entry_fill(self, signal: TrackedSignal):
        """Check if entry order has been filled."""
        if not signal.exchange_entry_order_id:
            return
        
        try:
            order = await self.exchange.get_order_by_id(signal.symbol, signal.exchange_entry_order_id)
            
            if order and order.get("orderStatus") == "Filled":
                # Entry filled!
                fill_price = float(order.get("avgPrice", signal.entry_price))
                fill_time = datetime.now(timezone.utc)
                
                # Update signal record
                await self.store.mark_as_filled(signal.signal_id, fill_price, fill_time)
                
                # Place initial stop loss and TP1
                await self._place_initial_exits(signal, fill_price)
                
                self._signal_states[signal.signal_id] = LifecycleState.ENTRY_FILLED
                logger.info(f"Entry filled for {signal.signal_id[:8]} @ {fill_price}")
            
            elif order and order.get("orderStatus") in ["Cancelled", "Rejected"]:
                logger.warning(f"Entry order {signal.exchange_entry_order_id} was {order.get('orderStatus')}")
                self._signal_states[signal.signal_id] = LifecycleState.CANCELLED
        
        except Exception as e:
            logger.error(f"Error checking entry fill: {e}")
    
    async def _place_initial_exits(self, signal: TrackedSignal, fill_price: float):
        """Place initial Stop Loss and TP1 orders after entry fill."""
        if not settings.AUTO_EXECUTE:
            return
        
        # Get position size from earlier calculation or re-calculate
        # For simplicity, assume we have the quantity stored
        # In production, fetch from exchange position
        position = await self.exchange.get_position(signal.symbol)
        if not position:
            logger.error(f"No position found for {signal.signal_id[:8]} after fill")
            return
        
        position_size = float(position.get("size", 0))
        half_position = position_size / 2.0
        
        side = OrderSide.SELL if signal.direction == "LONG" else OrderSide.BUY
        
        # Place TP1 order (50% at 1R)
        try:
            tp1_response = await self.exchange.place_limit_order(
                symbol=signal.symbol,
                side=side,
                price=signal.take_profit_1,
                qty=half_position,
                reduce_only=True,
                time_in_force="GTC",
            )
            signal.exchange_tp1_order_id = tp1_response.get("orderId")
            logger.info(f"TP1 order placed: {signal.symbol} {side.value} {half_position}@{signal.take_profit_1}")
        
        except Exception as e:
            logger.error(f"Failed to place TP1 order: {e}")
        
        # Place initial Stop Loss (100% position, will be reduced after TP1)
        try:
            sl_response = await self.exchange.place_limit_order(
                symbol=signal.symbol,
                side=side,
                price=signal.stop_loss,
                qty=position_size,
                reduce_only=True,
                time_in_force="GTC",
            )
            signal.exchange_sl_order_id = sl_response.get("orderId")
            logger.info(f"Initial SL order placed: {signal.symbol} {side.value} {position_size}@{signal.stop_loss}")
        
        except Exception as e:
            logger.error(f"Failed to place SL order: {e}")
        
        # Save order IDs
        await self.store.save_signal(signal)
    
    async def _monitor_active_position(self, signal: TrackedSignal):
        """Monitor active position for TP1 hit, SL hit, or MAE exit."""
        # Check if TP1 was hit
        if signal.tracking_status == TrackingStatus.HIT_TP1:
            # Transition to TP1 management
            await self._handle_tp1_hit(signal)
            return
        
        # Check for MAE early exit conditions
        if signal.mae_r is not None:
            # MAE > 1.0R and PnL < 0 → EARLY_EXIT_MAE
            if signal.mae_r > settings.MAE_HARD_EXIT_R:
                current_pnl_r = signal.calculate_pnl_r(signal.last_mark_price or signal.entry_price)
                if current_pnl_r < 0:
                    logger.warning(
                        f"MAE early exit triggered for {signal.signal_id[:8]}: "
                        f"MAE={signal.mae_r:.2f}R, PnL={current_pnl_r:.2f}R"
                    )
                    await self._emergency_close_position(signal, "EARLY_EXIT_MAE")
                    return
            
            # MAE > 0.7R and no +0.5R profit after 6 bars → EARLY_EXIT_STALL
            if signal.mae_r > settings.MAE_STALL_EXIT_R and signal.bars_without_profit >= settings.MAX_BARS_WITHOUT_PROFIT:
                logger.warning(
                    f"Stall exit triggered for {signal.signal_id[:8]}: "
                    f"MAE={signal.mae_r:.2f}R, bars_without_profit={signal.bars_without_profit}"
                )
                await self._emergency_close_position(signal, "EARLY_EXIT_STALL")
                return
        
        # Check for SL hit (would be handled by exchange order, but monitor for safety)
        current_price = signal.last_mark_price
        if current_price:
            if signal.direction == "LONG" and current_price <= signal.stop_loss:
                logger.info(f"SL hit for {signal.signal_id[:8]} @ {current_price}")
                # Exchange should handle this automatically via reduce-only order
            elif signal.direction == "SHORT" and current_price >= signal.stop_loss:
                logger.info(f"SL hit for {signal.signal_id[:8]} @ {current_price}")
    
    async def _handle_tp1_hit(self, signal: TrackedSignal):
        """Handle TP1 hit: move SL to +0.5R, place TP2 for runner."""
        if not settings.AUTO_EXECUTE:
            return
        
        # Calculate +0.5R stop price (avoids exact breakeven wick-outs)
        risk_amount = abs(signal.entry_price - signal.stop_loss)
        if signal.direction == "LONG":
            new_stop_price = signal.entry_price + (settings.BE_OFFSET_R * risk_amount)
        else:
            new_stop_price = signal.entry_price - (settings.BE_OFFSET_R * risk_amount)
        
        logger.info(
            f"TP1 hit for {signal.signal_id[:8]} - moving SL to +0.5R: "
            f"{signal.stop_loss:.2f} → {new_stop_price:.2f}"
        )
        
        # Cancel original SL
        if signal.exchange_sl_order_id:
            try:
                await self.exchange.cancel_order(signal.symbol, signal.exchange_sl_order_id)
            except Exception as e:
                logger.error(f"Failed to cancel original SL: {e}")
        
        # Place new SL at +0.5R for remaining 50%
        side = OrderSide.SELL if signal.direction == "LONG" else OrderSide.BUY
        
        # Get current position size (should be ~50% after TP1)
        position = await self.exchange.get_position(signal.symbol)
        if position:
            remaining_size = float(position.get("size", 0))
            
            try:
                sl_response = await self.exchange.place_limit_order(
                    symbol=signal.symbol,
                    side=side,
                    price=new_stop_price,
                    qty=remaining_size,
                    reduce_only=True,
                    time_in_force="GTC",
                )
                signal.exchange_sl_order_id = sl_response.get("orderId")
                signal.trailing_stop_price = new_stop_price
                await self.store.save_signal(signal)
                logger.info(f"New SL placed at +0.5R: {new_stop_price:.2f}")
            
            except Exception as e:
                logger.error(f"Failed to place new SL: {e}")
        
        # Place TP2 order (or set up trailing)
        # For simplicity, place limit at TP2
        if signal.take_profit_2:
            try:
                tp2_response = await self.exchange.place_limit_order(
                    symbol=signal.symbol,
                    side=side,
                    price=signal.take_profit_2,
                    qty=remaining_size,
                    reduce_only=True,
                    time_in_force="GTC",
                )
                signal.exchange_tp2_order_id = tp2_response.get("orderId")
                logger.info(f"TP2 order placed: {remaining_size}@{signal.take_profit_2}")
            
            except Exception as e:
                logger.error(f"Failed to place TP2 order: {e}")
        
        self._signal_states[signal.signal_id] = LifecycleState.TP2_ACTIVE
    
    async def _monitor_runner_position(self, signal: TrackedSignal):
        """Monitor runner position after TP1 hit (trailing stop logic)."""
        # Similar to _monitor_active_position but for remaining 50%
        # Would implement dynamic trailing based on ATR or highest/lowest price
        pass
    
    async def _check_trail_exit(self, signal: TrackedSignal):
        """Check if trailing stop was hit for runner position."""
        # Monitor for TP2 hit or trail exit
        pass
    
    async def _emergency_close_position(self, signal: TrackedSignal, reason: str):
        """Emergency market close for MAE/stall exits."""
        if not settings.AUTO_EXECUTE:
            logger.info(f"[PAPER MODE] Would emergency close {signal.signal_id[:8]}: {reason}")
            await self.store.update_tracking_status(
                signal.signal_id,
                TrackingStatus.EARLY_EXIT_MAE if "MAE" in reason else TrackingStatus.EARLY_EXIT_STALL,
                exit_price=signal.last_mark_price,
                exit_reason=reason,
            )
            return
        
        try:
            position = await self.exchange.get_position(signal.symbol)
            if position:
                size = float(position.get("size", 0))
                side = OrderSide.SELL if signal.direction == "LONG" else OrderSide.BUY
                
                await self.exchange.place_market_order(
                    symbol=signal.symbol,
                    side=side,
                    qty=size,
                    reduce_only=True,
                )
                
                logger.info(f"Emergency close executed for {signal.signal_id[:8]}: {reason}")
                
                # Update tracking status
                await self.store.update_tracking_status(
                    signal.signal_id,
                    TrackingStatus.EARLY_EXIT_MAE if "MAE" in reason else TrackingStatus.EARLY_EXIT_STALL,
                    exit_reason=reason,
                )
        
        except Exception as e:
            logger.error(f"Emergency close failed: {e}")


# Singleton instance
_lifecycle_manager: Optional[OrderLifecycleManager] = None


def get_lifecycle_manager() -> OrderLifecycleManager:
    """Get or create lifecycle manager singleton."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = OrderLifecycleManager()
    return _lifecycle_manager
