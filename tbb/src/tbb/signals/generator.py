from datetime import datetime, timedelta
from typing import Optional
import polars as pl
from tbb.core.config import settings
from tbb.signals.schema import TradeSignal, SignalDirection

class SignalGenerator:
    """
    Constructs TradeSignal objects when confluence threshold is met.
    
    Entry: 50% CE of active FVG (priority) or OB midpoint (fallback)
    Stop Loss: Beyond structural swing/sweep + 0.1*ATR buffer
    Take Profits: TP1 = 1.0R, TP2 = 2.5R
    Expiry: Dynamic based on timeframe
    """
    
    def __init__(self, config: dict = None):
        self.config = config if config else {}
    
    def generate_signal(
        self, 
        df: pl.DataFrame, 
        confluence_data: dict, 
        symbol: str, 
        timestamp: datetime
    ) -> Optional[TradeSignal]:
        """Generate a TradeSignal if conditions are met."""
        
        if len(df) == 0:
            return None
            
        last = df.tail(1)
        
        # Determine Direction from signals
        direction = self._determine_direction(last)
        if direction is None:
            return None
        
        # Calculate Entry Price
        entry_price = self._calculate_entry(last, direction)
        if entry_price is None:
            return None
        
        # Calculate Stop Loss
        stop_loss = self._calculate_stop(last, direction, entry_price)
        if stop_loss is None:
            return None
        
        # Calculate Risk and Take Profits
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return None
        
        tp1, tp2 = self._calculate_take_profits(entry_price, risk, direction)
        
        # Calculate Expiry
        expiry = self._calculate_expiry(timestamp)
        
        # Construct TradeSignal
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            confluence_score=confluence_data['score'],
            contributing_factors=confluence_data.get('factors', []),
            created_at=timestamp,
            expiry=expiry,
            status="PENDING"
        )
        
        return signal
    
    def _determine_direction(self, last: pl.DataFrame) -> Optional[SignalDirection]:
        """Determine trade direction from market structure signals."""
        is_high_sweep = last['is_high_sweep'][0] if 'is_high_sweep' in last.columns else False
        is_low_sweep = last['is_low_sweep'][0] if 'is_low_sweep' in last.columns else False
        is_bullish_bos = last['is_bullish_bos'][0] if 'is_bullish_bos' in last.columns else False
        is_bearish_bos = last['is_bearish_bos'][0] if 'is_bearish_bos' in last.columns else False
        
        if is_high_sweep or is_bullish_bos:
            return SignalDirection.LONG
        elif is_low_sweep or is_bearish_bos:
            return SignalDirection.SHORT
        
        return None
    
    def _calculate_entry(self, last: pl.DataFrame, direction: SignalDirection) -> Optional[float]:
        """Calculate entry price: FVG CE priority, then OB midpoint."""
        
        if direction == SignalDirection.LONG:
            # Priority 1: Bullish FVG CE
            if 'is_bullish_fvg_active' in last.columns and last['is_bullish_fvg_active'][0]:
                ce = last['bullish_fvg_ce'][0] if 'bullish_fvg_ce' in last.columns else None
                if ce is not None and ce > 0:
                    return ce
            
            # Fallback: Bullish OB Midpoint
            ob_top = last['bullish_ob_top'][0] if 'bullish_ob_top' in last.columns else None
            ob_bot = last['bullish_ob_bottom'][0] if 'bullish_ob_bottom' in last.columns else None
            
            if ob_top is not None and ob_bot is not None and ob_top > ob_bot:
                return (ob_top + ob_bot) / 2
                
        else:  # SHORT
            # Priority 1: Bearish FVG CE
            if 'is_bearish_fvg_active' in last.columns and last['is_bearish_fvg_active'][0]:
                ce = last['bearish_fvg_ce'][0] if 'bearish_fvg_ce' in last.columns else None
                if ce is not None and ce > 0:
                    return ce
            
            # Fallback: Bearish OB Midpoint
            ob_top = last['bearish_ob_top'][0] if 'bearish_ob_top' in last.columns else None
            ob_bot = last['bearish_ob_bottom'][0] if 'bearish_ob_bottom' in last.columns else None
            
            if ob_top is not None and ob_bot is not None and ob_top > ob_bot:
                return (ob_top + ob_bot) / 2
        
        return None
    
    def _calculate_stop(self, last: pl.DataFrame, direction: SignalDirection, entry: float) -> Optional[float]:
        """Calculate stop loss beyond structural level + 0.1*ATR buffer."""
        
        atr = last['atr_14'][0] if 'atr_14' in last.columns else 0
        buffer = 0.1 * atr if atr > 0 else entry * 0.001  # Fallback 0.1% if no ATR
        
        if direction == SignalDirection.LONG:
            # Find recent Swing Low or Sweep Low
            swing_low = last['swing_low_price'][0] if 'swing_low_price' in last.columns else None
            sweep_low = last['low'][0] if ('is_low_sweep' in last.columns and last['is_low_sweep'][0]) else None
            
            structural_level = None
            if swing_low and sweep_low:
                structural_level = min(swing_low, sweep_low)
            elif swing_low:
                structural_level = swing_low
            elif sweep_low:
                structural_level = sweep_low
            
            if structural_level is None or structural_level <= 0:
                return None
            
            # Stop must be BELOW structural level (for longs)
            return structural_level - buffer
            
        else:  # SHORT
            # Find recent Swing High or Sweep High
            swing_high = last['swing_high_price'][0] if 'swing_high_price' in last.columns else None
            sweep_high = last['high'][0] if ('is_high_sweep' in last.columns and last['is_high_sweep'][0]) else None
            
            structural_level = None
            if swing_high and sweep_high:
                structural_level = max(swing_high, sweep_high)
            elif swing_high:
                structural_level = swing_high
            elif sweep_high:
                structural_level = sweep_high
            
            if structural_level is None or structural_level <= 0:
                return None
            
            # Stop must be ABOVE structural level (for shorts)
            return structural_level + buffer
    
    def _calculate_take_profits(self, entry: float, risk: float, direction: SignalDirection) -> tuple:
        """Calculate TP1 (1.0R) and TP2 (2.5R)."""
        
        if direction == SignalDirection.LONG:
            tp1 = entry + (1.0 * risk)
            tp2 = entry + (2.5 * risk)
        else:  # SHORT
            tp1 = entry - (1.0 * risk)
            tp2 = entry - (2.5 * risk)
        
        return tp1, tp2
    
    def _calculate_expiry(self, timestamp: datetime) -> datetime:
        """Calculate signal expiry based on timeframe."""
        
        tf = self.config.get('timeframe', '15m')
        
        if tf in ['1m', '3m', '5m']:
            return timestamp + timedelta(hours=4)
        elif tf in ['15m', '30m', '1h']:
            return timestamp + timedelta(hours=24)
        elif tf == '4h':
            return timestamp + timedelta(hours=48)
        else:  # Daily or higher
            return timestamp + timedelta(days=7)
