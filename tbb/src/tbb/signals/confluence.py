"""
Confluence Scoring Engine - Merged Reversal Structure Pillar
Max Score: 0.70 (Reversal 0.30 + Retracement 0.25 + Displacement 0.15)
"""
import polars as pl
from tbb.core.config import settings


def score_confluence(df: pl.DataFrame, config: dict = None) -> dict | None:
    """
    Evaluate the latest closed candle for confluence of signals.
    Returns dict with score and factors if >= MIN_CONFLUENCE_SCORE, else None.
    """
    if config is None:
        config = {}
    
    if len(df) == 0:
        return None
        
    last = df.tail(1)
    min_score = config.get('min_confluence_score', settings.MIN_CONFLUENCE_SCORE)
    
    score = 0.0
    factors = []
    
    # === REVERSAL STRUCTURE PILLAR (+0.30 max) ===
    # Merged Sweep + BOS due to 72% co-firing correlation (not independent)
    is_sweep = last['is_high_sweep'][0] or last['is_low_sweep'][0]
    is_bos = last['is_bullish_bos'][0] or last['is_bearish_bos'][0]
    
    if is_sweep or is_bos:
        score += 0.30
        if is_sweep and is_bos:
            factors.append("Sweep+BOS Combo")
        elif is_sweep:
            factors.append("Liquidity Sweep")
        else:
            factors.append("Break of Structure")
    
    # === RETRACEMENT PILLAR (+0.25) ===
    if _check_in_fvg(last) or _check_in_ob(last):
        score += 0.25
        if _check_in_fvg(last):
            factors.append("FVG Retracement")
        if _check_in_ob(last):
            factors.append("OB Retracement")
    
    # === DISPLACEMENT PILLAR (+0.15) ===
    if _check_displacement(last, config):
        score += 0.15
        factors.append("Displacement")
    
    # Return if threshold met (max possible = 0.70)
    if score >= min_score:
        return {"score": score, "factors": factors}
    
    return None


def _check_in_fvg(last_row: pl.DataFrame) -> bool:
    """Check if price is retracing into an active FVG zone (near CE)."""
    close = last_row['close'][0]
    
    # Check Bullish FVG
    bull_active = last_row['is_bullish_fvg_active'][0] if 'is_bullish_fvg_active' in last_row.columns else False
    if bull_active:
        bull_ce = last_row['bullish_fvg_ce'][0] if 'bullish_fvg_ce' in last_row.columns else None
        if bull_ce is not None:
            # Price at or slightly above CE (within 0.5%)
            if close <= bull_ce * 1.005:
                return True
    
    # Check Bearish FVG
    bear_active = last_row['is_bearish_fvg_active'][0] if 'is_bearish_fvg_active' in last_row.columns else False
    if bear_active:
        bear_ce = last_row['bearish_fvg_ce'][0] if 'bearish_fvg_ce' in last_row.columns else None
        if bear_ce is not None:
            # Price at or slightly below CE (within 0.5%)
            if close >= bear_ce * 0.995:
                return True
    
    return False


def _check_in_ob(last_row: pl.DataFrame) -> bool:
    """Check if price is retracing into an Order Block zone."""
    close = last_row['close'][0]
    
    # Check Bullish OB
    bull_top = last_row['bullish_ob_top'][0] if 'bullish_ob_top' in last_row.columns else None
    bull_bot = last_row['bullish_ob_bottom'][0] if 'bullish_ob_bottom' in last_row.columns else None
    
    if bull_top is not None and bull_bot is not None:
        if bull_bot <= close <= bull_top:
            return True
    
    # Check Bearish OB
    bear_top = last_row['bearish_ob_top'][0] if 'bearish_ob_top' in last_row.columns else None
    bear_bot = last_row['bearish_ob_bottom'][0] if 'bearish_ob_bottom' in last_row.columns else None
    
    if bear_top is not None and bear_bot is not None:
        if bear_bot <= close <= bear_top:
            return True
    
    return False


def _check_displacement(last_row: pl.DataFrame, config: dict) -> bool:
    """Check for displacement via Volume or Large FVG Body."""
    vol = last_row['volume'][0] if 'volume' in last_row.columns else 0
    atr = last_row['atr_14'][0] if 'atr_14' in last_row.columns else 0
    
    # Volume Check: Volume > BOS_VOLUME_MULT * ATR-based expectation
    # For simplicity, check if volume is significantly above average (if available)
    vol_ma = last_row.get('volume_20_ma', [None])[0] if 'volume_20_ma' in last_row.columns else None
    bos_vol_mult = config.get('bos_volume_mult', settings.BOS_VOLUME_MULT)
    
    if vol_ma and vol > (vol_ma * bos_vol_mult):
        return True
    
    # Large FVG Body Check (Gap > FVG_GAP_ATR_MULT * ATR)
    # This requires pre-calculated FVG size column or derivation from OHLC
    # For MVP, rely primarily on volume. If specific FVG size exists:
    fvg_gap_mult = config.get('fvg_gap_atr_mult', settings.FVG_GAP_ATR_MULT)
    if 'fvg_size' in last_row.columns and last_row['fvg_size'][0] is not None:
        if last_row['fvg_size'][0] > (fvg_gap_mult * atr):
            return True
    
    # Fallback: If we have an active FVG, assume some displacement occurred
    bull_active = last_row.get('is_bullish_fvg_active', [False])[0] if 'is_bullish_fvg_active' in last_row.columns else False
    bear_active = last_row.get('is_bearish_fvg_active', [False])[0] if 'is_bearish_fvg_active' in last_row.columns else False
    
    if bull_active or bear_active:
        return True
    
    return False
