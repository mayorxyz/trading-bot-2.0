from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import polars as pl

from tbb.core.config import settings
from tbb.signals.schema import TradeSignal, SignalDirection


class SignalGenerator:
    """
    Constructs TradeSignal objects when confluence threshold is met.
    Entry: 50% CE of active FVG, or OB price.
    Stop Loss: Beyond structural swing/sweep + 0.1*ATR buffer.
    TP1: 1.0R, TP2: 2.5R.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_signal(
        self,
        df: pl.DataFrame,
        confluence_data: Dict[str, Any],
        symbol: str,
        timestamp: datetime
    ) -> Optional[TradeSignal]:
        """Generate TradeSignal if confluence threshold met."""
        if df.is_empty():
            return None
            
        last = df.tail(1)
        
        # Determine direction from sweep/BOS
        is_long = bool(last["is_high_sweep"][0] or last["is_bullish_bos"][0])
        is_short = bool(last["is_low_sweep"][0] or last["is_bearish_bos"][0])
        
        if not is_long and not is_short:
            return None
        
        direction = SignalDirection.LONG if is_long else SignalDirection.SHORT
        
        # Calculate entry price
        entry_price = self._calculate_entry(last, direction)
        if entry_price is None:
            return None
        
        # Calculate stop loss (structural + buffer)
        stop_loss = self._calculate_stop(last, direction, entry_price)
        if stop_loss is None:
            return None
        
        # Calculate risk and take profits
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return None
        
        if direction == SignalDirection.LONG:
            tp1 = entry_price + (1.0 * risk)
            tp2 = entry_price + (2.5 * risk)
        else:
            tp1 = entry_price - (1.0 * risk)
            tp2 = entry_price - (2.5 * risk)
        
        # Calculate expiry
        expiry = self._calculate_expiry(timestamp)
        
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            confluence_score=confluence_data["score"],
            contributing_factors=confluence_data["factors"],
            created_at=timestamp,
            expiry=expiry,
            status="PENDING"
        )
    
    def _calculate_entry(self, last: pl.DataFrame, direction: SignalDirection) -> Optional[float]:
        """Entry: 50% CE of FVG (priority) or OB midpoint (fallback)."""
        if direction == SignalDirection.LONG:
            if last["is_bullish_fvg_active"][0]:
                ce = last["bullish_fvg_ce"][0]
                if ce is not None:
                    return ce
            ob_top = last["bullish_ob_top"][0]
            ob_bot = last["bullish_ob_bottom"][0]
            if ob_top is not None and ob_bot is not None:
                return (ob_top + ob_bot) / 2.0
        else:
            if last["is_bearish_fvg_active"][0]:
                ce = last["bearish_fvg_ce"][0]
                if ce is not None:
                    return ce
            ob_top = last["bearish_ob_top"][0]
            ob_bot = last["bearish_ob_bottom"][0]
            if ob_top is not None and ob_bot is not None:
                return (ob_top + ob_bot) / 2.0
        
        return None
    
    def _calculate_stop(self, last: pl.DataFrame, direction: SignalDirection, entry: float) -> Optional[float]:
        """Stop: Beyond structural level (swing/sweep) + 0.1*ATR buffer."""
        atr = last["atr_14"][0] if "atr_14" in last.columns else 0.0
        buffer = 0.1 * atr if atr > 0 else entry * 0.001
        
        if direction == SignalDirection.LONG:
            swing_low = last["swing_low_price"][0]
            sweep_low = last["low"][0] if last["is_low_sweep"][0] else None
            levels = [l for l in [swing_low, sweep_low] if l is not None]
            if not levels:
                return None
            structural_level = min(levels)
            return structural_level - buffer
        else:
            swing_high = last["swing_high_price"][0]
            sweep_high = last["high"][0] if last["is_high_sweep"][0] else None
            levels = [l for l in [swing_high, sweep_high] if l is not None]
            if not levels:
                return None
            structural_level = max(levels)
            return structural_level + buffer
    
    def _calculate_expiry(self, timestamp: datetime) -> datetime:
        """Dynamic expiry based on timeframe."""
        tf = self.config.get("timeframe", "15m")
        
        if tf in ["1m", "2m", "3m", "5m"]:
            return timestamp + timedelta(hours=4)
        elif tf in ["15m", "30m", "1h"]:
            return timestamp + timedelta(hours=24)
        elif tf == "4h":
            return timestamp + timedelta(hours=48)
        else:
            return timestamp + timedelta(days=7)
