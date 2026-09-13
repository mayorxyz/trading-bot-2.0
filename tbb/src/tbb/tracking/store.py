"""
Signal Tracking Database Layer - Persists signals and tracks resolution.
Uses SQLite with SQLModel for ORM.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid

from sqlmodel import SQLModel, Field, create_engine, Session, select
from tbb.core.paths import get_tracking_db_path
from tbb.core.logger import setup_logger

logger = setup_logger(__name__)


class TrackingStatus(str, Enum):
    """Resolution status of a tracked signal."""
    PENDING = "pending"  # Signal published, waiting for fill
    ACTIVE = "active"  # Position filled, trade in progress
    HIT_TP1 = "hit_tp1"  # Hit first take profit (partial exit if configured)
    WIN_TP2_HARD = "win_tp2_hard"  # Hit TP2 hard (full win at fixed target)
    WIN_TP2_TRAILED = "win_tp2_trailed"  # Trailing stop hit on runner (partial win)
    LOSS_SL = "loss_sl"  # Hit stop loss
    EARLY_EXIT_MAE = "early_exit_mae"  # Early exit due to high MAE
    EARLY_EXIT_STALL = "early_exit_stall"  # Early exit due to stall (no profit after N bars)
    EXPIRED = "expired"  # Signal expired without fill
    INVALIDATED = "invalidated"  # Invalidated by opposite structure break
    CANCELLED = "cancelled"  # Manually cancelled
    TIME_EXIT = "time_exit"  # Time-based exit (48h timeout)


class TrackedSignal(SQLModel, table=True):
    """Database model for tracked signals with all resolution fields."""
    
    __tablename__ = "tracked_signals"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    signal_id: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # === Signal Details (copied from TradeSignal) ===
    symbol: str = Field(index=True)
    direction: str  # LONG | SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    confluence_score: float
    market_regime: str
    funding_rate: float
    atr_14: Optional[float]
    max_age_bars: int
    expiry_time: datetime
    
    # === MVS Conditions ===
    mvs_ob_present: bool
    mvs_fvg_confirmed: bool
    mvs_mss_confirmed: bool
    
    # === Tracking Fields ===
    tracking_status: TrackingStatus = Field(default=TrackingStatus.PENDING)
    entry_fill_price: Optional[float] = None
    entry_filled_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_r: Optional[float] = None
    pnl_pct: Optional[float] = None
    partial_pnl_r: Optional[float] = None  # PnL from 50% closed at TP1
    runner_pnl_r: Optional[float] = None   # PnL from remaining 50%
    total_pnl_r: Optional[float] = None    # Combined PnL
    mae_r: Optional[float] = None
    mfe_r: Optional[float] = None
    resolved_at: Optional[datetime] = None
    time_to_fill_seconds: Optional[int] = None
    time_to_resolution_seconds: Optional[int] = None
    time_to_tp1_seconds: Optional[int] = None
    
    # === Config at Signal Time ===
    move_sl_to_be_on_tp1: bool = Field(default=False)
    resolve_fully_at_tp1: bool = Field(default=False)
    trail_after_tp1: bool = Field(default=True)
    be_offset_r: float = Field(default=0.5)
    trail_atr_multiplier: float = Field(default=3.0)
    mae_hard_exit_r: float = Field(default=1.0)
    mae_stall_exit_r: float = Field(default=0.7)
    invalidation_reason: Optional[str] = None
    
    # === Price Monitoring ===
    highest_price_since_entry: Optional[float] = None
    lowest_price_since_entry: Optional[float] = None
    last_mark_price: Optional[float] = None
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # === Bars Monitoring ===
    bars_since_entry: int = Field(default=0)
    max_bars_for_resolution: int = Field(default=192)
    bars_without_profit: int = Field(default=0)
    
    # === Trailing Stop State ===
    trailing_stop_price: Optional[float] = None
    tp1_hit_at: Optional[datetime] = None
    position_remaining_pct: float = Field(default=1.0)  # 1.0 = full, 0.5 = half after TP1
    
    def calculate_pnl_r(self, exit_price: float) -> float:
        """Calculate PnL in R-multiples."""
        if self.direction == "LONG":
            pnl_points = exit_price - self.entry_price
            risk_points = self.entry_price - self.stop_loss
        else:
            pnl_points = self.stop_loss - exit_price
            risk_points = self.stop_loss - self.entry_price
        
        if risk_points <= 0:
            return 0.0
        
        return pnl_points / risk_points
    
    def calculate_pnl_pct(self, exit_price: float, leverage: int = 10) -> float:
        """Calculate PnL as percentage."""
        if self.direction == "LONG":
            pct_change = (exit_price - self.entry_price) / self.entry_price
        else:
            pct_change = (self.entry_price - exit_price) / self.entry_price
        
        return pct_change * leverage * 100
    
    def update_mae_mfe(self, current_high: float, current_low: float):
        """Update MAE and MFE based on current candle high/low."""
        if self.direction == "LONG":
            if self.mfe_r is None or current_high > self.highest_price_since_entry:
                self.highest_price_since_entry = current_high
                if self.entry_price != self.stop_loss:
                    self.mfe_r = (current_high - self.entry_price) / (self.entry_price - self.stop_loss)
            
            if self.mae_r is None or current_low < self.lowest_price_since_entry:
                self.lowest_price_since_entry = current_low
                if self.entry_price != self.stop_loss:
                    self.mae_r = (self.entry_price - current_low) / (self.entry_price - self.stop_loss)
        else:
            if self.mfe_r is None or current_low < self.lowest_price_since_entry:
                self.lowest_price_since_entry = current_low
                if self.stop_loss != self.entry_price:
                    self.mfe_r = (self.entry_price - current_low) / (self.stop_loss - self.entry_price)
            
            if self.mae_r is None or current_high > self.highest_price_since_entry:
                self.highest_price_since_entry = current_high
                if self.stop_loss != self.entry_price:
                    self.mae_r = (current_high - self.entry_price) / (self.stop_loss - self.entry_price)
    
    def calculate_trailing_stop(self, atr_14: Optional[float] = None, atr_multiplier: float = 3.0) -> float:
        """Calculate trailing stop price based on ATR or fixed percentage."""
        risk_amount = abs(self.entry_price - self.stop_loss)
        
        if self.direction == "LONG":
            if atr_14 and self.highest_price_since_entry:
                # Trail at highest_price - (ATR * multiplier)
                trail_distance = atr_14 * atr_multiplier
                return max(self.stop_loss, self.highest_price_since_entry - trail_distance)
            else:
                # Fallback: trail at +0.5R from entry
                return self.entry_price + (risk_amount * self.be_offset_r)
        else:  # SHORT
            if atr_14 and self.lowest_price_since_entry:
                trail_distance = atr_14 * atr_multiplier
                return min(self.stop_loss, self.lowest_price_since_entry + trail_distance)
            else:
                return self.entry_price - (risk_amount * self.be_offset_r)
    
    def calculate_be_offset_stop(self) -> float:
        """Calculate stop loss at +0.5R (avoids exact breakeven wick-outs)."""
        risk_amount = abs(self.entry_price - self.stop_loss)
        
        if self.direction == "LONG":
            return self.entry_price + (risk_amount * self.be_offset_r)
        else:  # SHORT
            return self.entry_price - (risk_amount * self.be_offset_r)
    
    def get_current_stop_loss(self) -> float:
        """Get current active stop loss (may have been trailed)."""
        if self.trailing_stop_price:
            return self.trailing_stop_price
        return self.stop_loss


class SignalTrackingStore:
    """Async database operations for signal tracking."""
    
    def __init__(self):
        self.db_path = get_tracking_db_path()
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        self._session_lock = asyncio.Lock()
        self._ensure_tables()
        logger.info(f"SignalTrackingStore initialized at {self.db_path}")
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
    
    def get_sync_session(self) -> Session:
        """Get synchronous session for simple operations."""
        return Session(self.engine)
    
    async def save_signal(self, signal: TrackedSignal) -> TrackedSignal:
        """Save or update a tracked signal."""
        async with self._session_lock:
            with Session(self.engine) as session:
                existing = session.exec(
                    select(TrackedSignal).where(TrackedSignal.signal_id == signal.signal_id)
                ).first()
                
                if existing:
                    for field in signal.__fields__:
                        value = getattr(signal, field)
                        if value is not None:
                            setattr(existing, field, value)
                    existing.last_update = datetime.now(timezone.utc)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    return existing
                else:
                    session.add(signal)
                    session.commit()
                    session.refresh(signal)
                    return signal
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[TrackedSignal]:
        """Get all active/pending signals."""
        async with self._session_lock:
            with Session(self.engine) as session:
                query = select(TrackedSignal).where(
                    TrackedSignal.tracking_status.in_([
                        TrackingStatus.PENDING,
                        TrackingStatus.ACTIVE,
                        TrackingStatus.HIT_TP1
                    ])
                )
                
                if symbol:
                    query = query.where(TrackedSignal.symbol == symbol)
                
                results = session.exec(query)
                return list(results.all())
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get a specific signal by ID."""
        async with self._session_lock:
            with Session(self.engine) as session:
                return session.exec(
                    select(TrackedSignal).where(TrackedSignal.signal_id == signal_id)
                ).first()
    
    async def get_resolved_signals(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrackedSignal]:
        """Get resolved signals for stats calculation."""
        async with self._session_lock:
            with Session(self.engine) as session:
                query = select(TrackedSignal).where(
                    TrackedSignal.tracking_status.in_([
                        TrackingStatus.WIN_TP2,
                        TrackingStatus.LOSS_SL,
                        TrackingStatus.EXPIRED,
                        TrackingStatus.INVALIDATED,
                    ])
                )
                
                if symbol:
                    query = query.where(TrackedSignal.symbol == symbol)
                
                if start_date:
                    query = query.where(TrackedSignal.resolved_at >= start_date)
                
                if end_date:
                    query = query.where(TrackedSignal.resolved_at <= end_date)
                
                query = query.order_by(TrackedSignal.resolved_at.desc()).limit(limit).offset(offset)
                results = session.exec(query)
                return list(results.all())
    
    async def update_tracking_status(
        self,
        signal_id: str,
        status: TrackingStatus,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> Optional[TrackedSignal]:
        """Update tracking status and optionally set exit details."""
        async with self._session_lock:
            with Session(self.engine) as session:
                signal = session.exec(
                    select(TrackedSignal).where(TrackedSignal.signal_id == signal_id)
                ).first()
                
                if not signal:
                    logger.warning(f"Signal {signal_id} not found for status update")
                    return None
                
                signal.tracking_status = status
                signal.last_update = datetime.now(timezone.utc)
                
                if exit_price:
                    signal.exit_price = exit_price
                    signal.pnl_r = signal.calculate_pnl_r(exit_price)
                    signal.pnl_pct = signal.calculate_pnl_pct(exit_price)
                
                if exit_reason:
                    signal.exit_reason = exit_reason
                
                if status in [TrackingStatus.WIN_TP2, TrackingStatus.LOSS_SL, 
                              TrackingStatus.EXPIRED, TrackingStatus.INVALIDATED]:
                    signal.resolved_at = datetime.now(timezone.utc)
                    
                    if signal.created_at:
                        signal.time_to_resolution_seconds = int(
                            (signal.resolved_at - signal.created_at).total_seconds()
                        )
                
                session.add(signal)
                session.commit()
                session.refresh(signal)
                
                logger.info(
                    f"Signal {signal_id[:8]} status updated to {status.value} | "
                    f"PnL: {signal.pnl_r:.2f}R" if signal.pnl_r else ""
                )
                
                return signal
    
    async def update_price_monitoring(
        self,
        signal_id: str,
        mark_price: float,
        candle_high: float,
        candle_low: float,
    ) -> Optional[TrackedSignal]:
        """Update price monitoring and MAE/MFE for an active signal."""
        async with self._session_lock:
            with Session(self.engine) as session:
                signal = session.exec(
                    select(TrackedSignal).where(TrackedSignal.signal_id == signal_id)
                ).first()
                
                if not signal or signal.tracking_status not in [
                    TrackingStatus.ACTIVE, TrackingStatus.HIT_TP1
                ]:
                    return None
                
                signal.last_mark_price = mark_price
                signal.bars_since_entry += 1
                signal.update_mae_mfe(candle_high, candle_low)
                signal.last_update = datetime.now(timezone.utc)
                
                session.add(signal)
                session.commit()
                session.refresh(signal)
                
                return signal
    
    async def mark_as_filled(
        self,
        signal_id: str,
        fill_price: float,
        fill_timestamp: datetime,
    ) -> Optional[TrackedSignal]:
        """Mark signal as filled (transition from PENDING to ACTIVE)."""
        async with self._session_lock:
            with Session(self.engine) as session:
                signal = session.exec(
                    select(TrackedSignal).where(TrackedSignal.signal_id == signal_id)
                ).first()
                
                if not signal:
                    return None
                
                signal.tracking_status = TrackingStatus.ACTIVE
                signal.entry_fill_price = fill_price
                signal.entry_filled_at = fill_timestamp
                signal.last_update = datetime.now(timezone.utc)
                
                signal.highest_price_since_entry = fill_price
                signal.lowest_price_since_entry = fill_price
                signal.last_mark_price = fill_price
                
                if signal.created_at and fill_timestamp:
                    signal.time_to_fill_seconds = int(
                        (fill_timestamp - signal.created_at).total_seconds()
                    )
                
                session.add(signal)
                session.commit()
                session.refresh(signal)
                
                logger.info(f"Signal {signal_id[:8]} marked as filled @ {fill_price}")
                return signal
    
    async def get_tracking_stats(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        min_confluence_score: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate tracking statistics for API endpoint with advanced metrics."""
        async with self._session_lock:
            with Session(self.engine) as session:
                # Include all resolved statuses including early exits
                query = select(TrackedSignal).where(
                    TrackedSignal.tracking_status.in_([
                        TrackingStatus.WIN_TP2_HARD,
                        TrackingStatus.WIN_TP2_TRAILED,
                        TrackingStatus.LOSS_SL,
                        TrackingStatus.EARLY_EXIT_MAE,
                        TrackingStatus.EARLY_EXIT_STALL,
                        TrackingStatus.EXPIRED,
                        TrackingStatus.INVALIDATED,
                        TrackingStatus.TIME_EXIT,
                    ])
                )
                
                if symbol:
                    query = query.where(TrackedSignal.symbol == symbol)
                
                if regime:
                    query = query.where(TrackedSignal.market_regime == regime)
                
                if min_confluence_score:
                    query = query.where(TrackedSignal.confluence_score >= min_confluence_score)
                
                if start_date:
                    query = query.where(TrackedSignal.resolved_at >= start_date)
                
                if end_date:
                    query = query.where(TrackedSignal.resolved_at <= end_date)
                
                results = session.exec(query)
                resolved_signals = list(results.all())
                
                if not resolved_signals:
                    return {
                        "total_signals": 0,
                        "fill_rate": None,
                        "win_rate": None,
                        "profit_factor": None,
                        "expectancy_r": None,
                        "avg_mae_r": None,
                        "avg_mfe_r": None,
                        "mae_median_winners": None,
                        "mfe_capture_rate": None,
                        "early_exit_rate": None,
                        "invalidation_rate": None,
                        "avg_time_to_fill_hours": None,
                        "avg_time_to_resolution_hours": None,
                        "avg_time_to_tp1_hours": None,
                    }
                
                total_count = len(resolved_signals)
                
                # Wins include TP2 hard, TP2 trailed, and any positive PnL exit
                wins = [s for s in resolved_signals if s.total_pnl_r and s.total_pnl_r > 0]
                if not wins:  # Fallback to pnl_r if total_pnl_r not set
                    wins = [s for s in resolved_signals if s.pnl_r and s.pnl_r > 0]
                
                losses = [s for s in resolved_signals if (s.total_pnl_r or s.pnl_r or 0) <= 0]
                
                win_rate = len(wins) / total_count if total_count > 0 else None
                
                # Use total_pnl_r if available, otherwise pnl_r
                gross_profit = sum(s.total_pnl_r or s.pnl_r or 0 for s in wins)
                gross_loss = abs(sum(s.total_pnl_r or s.pnl_r or 0 for s in losses))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
                
                expectancy_r = sum(s.total_pnl_r or s.pnl_r or 0 for s in resolved_signals) / total_count
                
                # MAE/MFE stats
                mae_signals = [s for s in resolved_signals if s.mae_r is not None]
                mfe_signals = [s for s in resolved_signals if s.mfe_r is not None]
                avg_mae_r = sum(s.mae_r for s in mae_signals) / len(mae_signals) if mae_signals else None
                avg_mfe_r = sum(s.mfe_r for s in mfe_signals) / len(mfe_signals) if mfe_signals else None
                
                # MAE median for winners
                winner_mae_values = [s.mae_r for s in wins if s.mae_r is not None]
                mae_median_winners = None
                if winner_mae_values:
                    sorted_mae = sorted(winner_mae_values)
                    mid = len(sorted_mae) // 2
                    mae_median_winners = sorted_mae[mid] if len(sorted_mae) % 2 else (sorted_mae[mid-1] + sorted_mae[mid]) / 2
                
                # MFE capture rate: Total PnL / MFE
                mfe_capture_sum = 0.0
                mfe_capture_count = 0
                for s in resolved_signals:
                    if s.mfe_r and s.mfe_r > 0 and (s.total_pnl_r or s.pnl_r or 0) > 0:
                        capture = (s.total_pnl_r or s.pnl_r) / s.mfe_r
                        mfe_capture_sum += capture
                        mfe_capture_count += 1
                mfe_capture_rate = mfe_capture_sum / mfe_capture_count if mfe_capture_count > 0 else None
                
                # Early exit rate
                early_exits = [s for s in resolved_signals if s.tracking_status in [
                    TrackingStatus.EARLY_EXIT_MAE, TrackingStatus.EARLY_EXIT_STALL
                ]]
                early_exit_rate = len(early_exits) / total_count if total_count > 0 else None
                
                # Invalidation rate
                invalidations = [s for s in resolved_signals if s.tracking_status == TrackingStatus.INVALIDATED]
                invalidation_rate = len(invalidations) / total_count if total_count > 0 else None
                
                # All signals for fill rate
                all_signals_query = select(TrackedSignal)
                if symbol:
                    all_signals_query = all_signals_query.where(TrackedSignal.symbol == symbol)
                all_signals = session.exec(all_signals_query).all()
                
                filled_count = len([s for s in all_signals if s.tracking_status in [
                    TrackingStatus.ACTIVE, TrackingStatus.HIT_TP1, 
                    TrackingStatus.WIN_TP2_HARD, TrackingStatus.WIN_TP2_TRAILED,
                    TrackingStatus.LOSS_SL, TrackingStatus.EARLY_EXIT_MAE,
                    TrackingStatus.EARLY_EXIT_STALL,
                ]])
                fill_rate = filled_count / len(all_signals) if all_signals else None
                
                # Time metrics
                fill_times = [s.time_to_fill_seconds for s in resolved_signals if s.time_to_fill_seconds]
                resolution_times = [s.time_to_resolution_seconds for s in resolved_signals if s.time_to_resolution_seconds]
                tp1_times = [s.time_to_tp1_seconds for s in resolved_signals if s.time_to_tp1_seconds]
                
                avg_time_to_fill = sum(fill_times) / len(fill_times) if fill_times else None
                avg_time_to_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else None
                avg_time_to_tp1 = sum(tp1_times) / len(tp1_times) if tp1_times else None
                
                return {
                    "total_signals": total_count,
                    "fill_rate": round(fill_rate, 3) if fill_rate else None,
                    "win_rate": round(win_rate, 3) if win_rate else None,
                    "profit_factor": round(profit_factor, 3) if profit_factor else None,
                    "expectancy_r": round(expectancy_r, 3) if expectancy_r else None,
                    "avg_mae_r": round(avg_mae_r, 3) if avg_mae_r else None,
                    "avg_mfe_r": round(avg_mfe_r, 3) if avg_mfe_r else None,
                    "mae_median_winners": round(mae_median_winners, 3) if mae_median_winners else None,
                    "mfe_capture_rate": round(mfe_capture_rate, 3) if mfe_capture_rate else None,
                    "early_exit_rate": round(early_exit_rate, 3) if early_exit_rate else None,
                    "invalidation_rate": round(invalidation_rate, 3) if invalidation_rate else None,
                    "avg_time_to_fill_hours": round(avg_time_to_fill / 3600, 2) if avg_time_to_fill else None,
                    "avg_time_to_resolution_hours": round(avg_time_to_resolution / 3600, 2) if avg_time_to_resolution else None,
                    "avg_time_to_tp1_hours": round(avg_time_to_tp1 / 3600, 2) if avg_time_to_tp1 else None,
                }


tracking_store = SignalTrackingStore()
