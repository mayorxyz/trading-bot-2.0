"""
Signal Output Module - Formats and delivers trade signals via Telegram and API.
Primary output of the bot is the signal itself; execution is optional.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid

from tbb.core.config import settings
from tbb.core.logger import setup_logger
from tbb.monitoring.alerts import TelegramAlerter
from tbb.tracking.store import tracking_store, TrackedSignal, TrackingStatus

logger = setup_logger(__name__)


class SignalStatus(Enum):
    """Lifecycle status of a trade signal."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    STOPPED = "stopped"
    TAKE_PROFIT = "take_profit"
    TIME_EXIT = "time_exit"


@dataclass
class TradeSignal:
    """Complete trade signal output - the PRIMARY product of the bot."""
    signal_id: str
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    confluence_score: float
    mvs_ob_present: bool
    mvs_fvg_confirmed: bool
    mvs_mss_confirmed: bool
    market_regime: str
    adx_value: Optional[float]
    hurst_value: Optional[float]
    funding_rate: float
    atr_14: Optional[float]
    cvd_divergence: bool
    obi_value: Optional[float]
    expiry_time: datetime
    max_age_bars: int
    bars_remaining: int = 0
    status: SignalStatus = SignalStatus.PENDING
    auto_execute: bool = False
    execution_triggered: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    position_size: Optional[float] = None
    timeframe_entry: str = "15m"
    timeframe_htf: str = "4H"
    invalidation_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response and logging."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_reward": round(self.risk_reward, 3),
            "confluence_score": round(self.confluence_score, 3),
            "mvs_conditions": {
                "ob_present": self.mvs_ob_present,
                "fvg_confirmed": self.mvs_fvg_confirmed,
                "mss_confirmed": self.mvs_mss_confirmed,
            },
            "market_context": {
                "regime": self.market_regime,
                "adx": round(self.adx_value, 2) if self.adx_value else None,
                "hurst": round(self.hurst_value, 3) if self.hurst_value else None,
                "funding_rate": self.funding_rate,
                "atr_14": round(self.atr_14, 4) if self.atr_14 else None,
                "cvd_divergence": self.cvd_divergence,
                "obi": round(self.obi_value, 3) if self.obi_value else None,
            },
            "validity": {
                "expiry_time": self.expiry_time.isoformat(),
                "max_age_bars": self.max_age_bars,
                "bars_remaining": self.bars_remaining,
            },
            "execution": {
                "status": self.status.value,
                "auto_execute": self.auto_execute,
                "execution_triggered": self.execution_triggered,
                "fill_price": self.fill_price,
                "fill_timestamp": self.fill_timestamp.isoformat() if self.fill_timestamp else None,
                "position_size": self.position_size,
            },
            "invalidation_reason": self.invalidation_reason,
            "notes": self.notes,
        }
    
    def format_telegram_message(self) -> str:
        """Format signal as Telegram message with emoji and structure."""
        direction_emoji = "🟢" if self.direction == "LONG" else "🔴"
        regime_emoji = "📈" if "TRENDING" in self.market_regime else "📊"
        conditions_met = sum([self.mvs_ob_present, self.mvs_fvg_confirmed, self.mvs_mss_confirmed])
        conditions_str = f"{conditions_met}/3 MVS conditions"
        score_color = "🟢" if self.confluence_score >= 0.6 else "🟡" if self.confluence_score >= 0.4 else "🔴"
        hours_until_expiry = (self.expiry_time - datetime.now(timezone.utc)).total_seconds() / 3600
        
        message = f"""
{direction_emoji} *NEW SIGNAL: {self.symbol}* {direction_emoji}

📊 *Entry Zone*
├ Entry: ${self.entry_price:,.2f}
├ Stop: ${self.stop_loss:,.2f}
├ TP1 (1R): ${self.take_profit_1:,.2f}
└ TP2: ${self.take_profit_2:,.2f}

📈 *Risk/Reward*
├ R:R Ratio: {self.risk_reward:.2f}
└ Confluence: {score_color} {self.confluence_score:.2f}

✅ *MVS Conditions* ({conditions_str})
├ HTF Order Block: {'✓' if self.mvs_ob_present else '✗'}
├ FVG Confirmation: {'✓' if self.mvs_fvg_confirmed else '✗'}
└ MSS/CHoCH: {'✓' if self.mvs_mss_confirmed else '✗'}

🌐 *Market Context* {regime_emoji}
├ Regime: {self.market_regime}
├ ADX: {self.adx_value:.1f if self.adx_value else 'N/A'}
├ Hurst: {self.hurst_value:.3f if self.hurst_value else 'N/A'}
├ Funding: {self.funding_rate*100:.4f}%
├ ATR(14): {self.atr_14:.4f if self.atr_14 else 'N/A'}
├ CVD Divergence: {'⚠️ Yes' if self.cvd_divergence else 'No'}
└ OBI: {self.obi_value:.3f if self.obi_value else 'N/A'}

⏰ *Validity Window*
├ Expires in: {hours_until_expiry:.1f} hours
├ Bars remaining: {self.bars_remaining}
└ Auto-Execute: {'🤖 ON' if self.auto_execute else '🔒 OFF'}

🆔 Signal ID: `{self.signal_id[:8]}...`
        """.strip()
        return message
    
    def is_valid(self) -> bool:
        """Check if signal is still valid for entry."""
        if self.status != SignalStatus.ACTIVE:
            return False
        if datetime.now(timezone.utc) > self.expiry_time:
            return False
        if self.bars_remaining <= 0:
            return False
        return True
    
    def should_auto_execute(self) -> bool:
        """Determine if auto-execution should trigger."""
        return (
            self.auto_execute 
            and self.is_valid() 
            and not self.execution_triggered
            and self.confluence_score >= 0.6
        )


class SignalOutputManager:
    """Manages signal delivery via Telegram and API endpoint."""
    
    def __init__(self):
        self.active_signals: Dict[str, TradeSignal] = {}
        self.signal_history: List[TradeSignal] = []
        self.telegram_alerter = TelegramAlerter() if settings.TELEGRAM_BOT_TOKEN else None
        self._lock = asyncio.Lock()
        logger.info("SignalOutputManager initialized")
    
    async def publish_signal(self, signal: TradeSignal) -> None:
        """Publish a new trade signal via all configured channels."""
        async with self._lock:
            signal.status = SignalStatus.ACTIVE
            signal.bars_remaining = signal.max_age_bars
            self.active_signals[signal.signal_id] = signal
            self.signal_history.append(signal)
            
            # Also persist to tracking database
            tracked_signal = TrackedSignal(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                risk_reward=signal.risk_reward,
                confluence_score=signal.confluence_score,
                market_regime=signal.market_regime,
                funding_rate=signal.funding_rate,
                atr_14=signal.atr_14,
                max_age_bars=signal.max_age_bars,
                expiry_time=signal.expiry_time,
                mvs_ob_present=signal.mvs_ob_present,
                mvs_fvg_confirmed=signal.mvs_fvg_confirmed,
                mvs_mss_confirmed=signal.mvs_mss_confirmed,
                move_sl_to_be_on_tp1=getattr(settings, 'MOVE_SL_TO_BE_ON_TP1', False),
                resolve_fully_at_tp1=getattr(settings, 'RESOLVE_FULLY_AT_TP1', False),
            )
            await tracking_store.save_signal(tracked_signal)
            
            logger.info(
                f"Signal published: {signal.symbol} {signal.direction} | "
                f"Entry: {signal.entry_price} | Score: {signal.confluence_score:.2f}"
            )
            
            if signal.confluence_score >= 0.6 and self.telegram_alerter:
                try:
                    await self.telegram_alerter.send_alert(
                        level="CRITICAL",
                        message=signal.format_telegram_message(),
                        category="new_signal"
                    )
                    logger.info(f"Telegram alert sent for signal {signal.signal_id[:8]}")
                except Exception as e:
                    logger.error(f"Failed to send Telegram alert: {e}")
            
            if signal.should_auto_execute():
                await self._trigger_execution(signal)
    
    async def _trigger_execution(self, signal: TradeSignal) -> None:
        """Trigger automated execution if AUTO_EXECUTE=true."""
        if not signal.auto_execute:
            return
        
        logger.warning(f"AUTO-EXECUTE triggered for signal {signal.signal_id[:8]}")
        signal.execution_triggered = True
        
        from tbb.execution.order import OrderExecutor
        executor = OrderExecutor()
        try:
            order_result = await executor.place_limit_order(
                symbol=signal.symbol,
                side=signal.direction,
                price=signal.entry_price,
                size=signal.position_size or 0.0,
                leverage=settings.DEFAULT_LEVERAGE,
                reduce_only=False,
            )
            
            if order_result.success:
                signal.status = SignalStatus.FILLED
                signal.fill_price = order_result.fill_price
                signal.fill_timestamp = datetime.now(timezone.utc)
                logger.info(f"Order filled: {signal.symbol} @ {order_result.fill_price}")
            else:
                signal.status = SignalStatus.PENDING
                signal.notes.append(f"Order placement failed: {order_result.message}")
        except Exception as e:
            signal.status = SignalStatus.PENDING
            signal.notes.append(f"Execution error: {str(e)}")
            logger.error(f"Execution error for signal {signal.signal_id[:8]}: {e}")
    
    async def update_signal_status(
        self,
        signal_id: str,
        status: SignalStatus,
        fill_price: Optional[float] = None,
        position_size: Optional[float] = None,
        invalidation_reason: Optional[str] = None,
    ) -> None:
        """Update signal status."""
        async with self._lock:
            if signal_id not in self.active_signals:
                logger.warning(f"Signal {signal_id} not found for status update")
                return
            
            signal = self.active_signals[signal_id]
            signal.status = status
            
            if fill_price:
                signal.fill_price = fill_price
            if position_size:
                signal.position_size = position_size
            if invalidation_reason:
                signal.invalidation_reason = invalidation_reason
                signal.status = SignalStatus.CANCELLED
            
            if status in [SignalStatus.EXPIRED, SignalStatus.CANCELLED, SignalStatus.FILLED]:
                del self.active_signals[signal_id]
            
            logger.info(f"Signal {signal_id[:8]} status updated to {status.value}")
    
    async def decrement_bars_remaining(self, symbol: str, timeframe: str = "1h") -> None:
        """Decrement bars_remaining for all active signals on symbol."""
        async with self._lock:
            for signal in list(self.active_signals.values()):
                if signal.symbol == symbol and signal.timeframe_entry == timeframe:
                    signal.bars_remaining -= 1
                    
                    if signal.bars_remaining <= 0:
                        signal.status = SignalStatus.EXPIRED
                        signal.invalidation_reason = "FVG age limit reached"
                        del self.active_signals[signal.signal_id]
                        logger.info(f"Signal {signal.signal_id[:8]} expired (age limit)")
                        
                        if self.telegram_alerter:
                            await self.telegram_alerter.send_alert(
                                level="WARNING",
                                message=f"⏰ Signal Expired: {signal.symbol} {signal.direction}\nID: `{signal.signal_id[:8]}`\nReason: FVG age limit",
                                category="signal_expiry"
                            )
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[TradeSignal]:
        """Get all active signals, optionally filtered by symbol."""
        signals = list(self.active_signals.values())
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        return sorted(signals, key=lambda s: s.timestamp, reverse=True)
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TradeSignal]:
        """Get paginated signal history for API endpoint."""
        signals = self.signal_history
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        return signals[offset:offset + limit]
    
    def get_signal_by_id(self, signal_id: str) -> Optional[TradeSignal]:
        """Get a specific signal by ID."""
        return self.active_signals.get(signal_id) or next(
            (s for s in self.signal_history if s.signal_id == signal_id),
            None
        )


signal_output_manager = SignalOutputManager()


async def publish_trade_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    risk_reward: float,
    confluence_score: float,
    mvs_ob_present: bool,
    mvs_fvg_confirmed: bool,
    mvs_mss_confirmed: bool,
    market_regime: str,
    adx_value: Optional[float],
    hurst_value: Optional[float],
    funding_rate: float,
    atr_14: Optional[float],
    cvd_divergence: bool,
    obi_value: Optional[float],
    max_age_bars: int,
    auto_execute: bool = False,
    position_size: Optional[float] = None,
    notes: Optional[List[str]] = None,
) -> TradeSignal:
    """Convenience function to create and publish a trade signal."""
    now = datetime.now(timezone.utc)
    hours_until_expiry = max_age_bars
    expiry_time = datetime.fromtimestamp(
        now.timestamp() + (hours_until_expiry * 3600),
        tz=timezone.utc
    )
    
    signal = TradeSignal(
        signal_id=str(uuid.uuid4()),
        timestamp=now,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_reward=risk_reward,
        confluence_score=confluence_score,
        mvs_ob_present=mvs_ob_present,
        mvs_fvg_confirmed=mvs_fvg_confirmed,
        mvs_mss_confirmed=mvs_mss_confirmed,
        market_regime=market_regime,
        adx_value=adx_value,
        hurst_value=hurst_value,
        funding_rate=funding_rate,
        atr_14=atr_14,
        cvd_divergence=cvd_divergence,
        obi_value=obi_value,
        expiry_time=expiry_time,
        max_age_bars=max_age_bars,
        auto_execute=auto_execute,
        position_size=position_size,
        notes=notes or [],
    )
    
    await signal_output_manager.publish_signal(signal)
    return signal
