"""
Tests for Confluence Scoring Engine
"""
import polars as pl
from datetime import datetime
from tbb.signals.confluence import score_confluence


def test_perfect_setup_scores_0_70():
    """
    Perfect setup (Sweep/BOS + FVG Retracement + Displacement) scores 0.70.
    """
    data = {
        'close': [100.0],
        'is_high_sweep': [True],  # Reversal Structure (+0.30)
        'is_low_sweep': [False],
        'is_bullish_bos': [False],
        'is_bearish_bos': [False],
        'is_bullish_fvg_active': [True],  # Retracement (+0.25)
        'bullish_fvg_ce': [100.0],
        'is_bearish_fvg_active': [False],
        'bearish_fvg_ce': [None],
        'bullish_ob_top': [None],
        'bullish_ob_bottom': [None],
        'bearish_ob_top': [None],
        'bearish_ob_bottom': [None],
        'volume': [5000.0],  # High volume for Displacement (+0.15)
        'volume_20_ma': [1000.0],
        'atr_14': [10.0]
    }
    df = pl.DataFrame(data)
    
    result = score_confluence(df, {'min_confluence_score': 0.40})
    
    assert result is not None
    assert result['score'] == 0.70
    assert "Liquidity Sweep" in result['factors']
    assert "FVG Retracement" in result['factors']
    assert "Displacement" in result['factors']


def test_weak_setup_returns_none():
    """
    Weak setup (single factor) scores below threshold and returns None.
    """
    data = {
        'close': [100.0],
        'is_high_sweep': [False],  # No reversal
        'is_low_sweep': [False],
        'is_bullish_bos': [False],
        'is_bearish_bos': [False],
        'is_bullish_fvg_active': [True],  # Only retracement (+0.25)
        'bullish_fvg_ce': [100.0],
        'is_bearish_fvg_active': [False],
        'bearish_fvg_ce': [None],
        'bullish_ob_top': [None],
        'bullish_ob_bottom': [None],
        'volume': [500.0],  # Low volume
        'volume_20_ma': [1000.0],
        'atr_14': [10.0]
    }
    df = pl.DataFrame(data)
    
    result = score_confluence(df, {'min_confluence_score': 0.40})
    
    assert result is None  # 0.25 < 0.40 threshold


def test_sweep_bos_merged_not_double_counted():
    """
    Verify Sweep+BOS combo still scores only 0.30 (not 0.30+0.20).
    """
    data = {
        'close': [100.0],
        'is_high_sweep': [True],  # Both sweep AND BOS
        'is_low_sweep': [False],
        'is_bullish_bos': [True],
        'is_bearish_bos': [False],
        'is_bullish_fvg_active': [False],  # No retracement
        'bullish_fvg_ce': [None],
        'is_bearish_fvg_active': [False],
        'bearish_fvg_ce': [None],
        'volume': [500.0],  # No displacement
        'volume_20_ma': [1000.0],
        'atr_14': [10.0]
    }
    df = pl.DataFrame(data)
    
    result = score_confluence(df, {'min_confluence_score': 0.20})
    
    assert result is not None
    assert result['score'] == 0.30  # NOT 0.50
    assert "Sweep+BOS Combo" in result['factors']


def test_retracement_in_ob():
    """
    Verify retracement scoring works for OB zones (not just FVG).
    """
    data = {
        'close': [95.0],  # Inside OB zone (90-100)
        'is_high_sweep': [True],  # Reversal (+0.30)
        'is_low_sweep': [False],
        'is_bullish_bos': [False],
        'is_bearish_bos': [False],
        'is_bullish_fvg_active': [False],  # No FVG
        'bullish_fvg_ce': [None],
        'bullish_ob_top': [100.0],  # OB present (+0.25)
        'bullish_ob_bottom': [90.0],
        'volume': [500.0],
        'volume_20_ma': [1000.0],
        'atr_14': [10.0]
    }
    df = pl.DataFrame(data)
    
    result = score_confluence(df, {'min_confluence_score': 0.40})
    
    assert result is not None
    assert result['score'] == 0.55  # 0.30 + 0.25
    assert "OB Retracement" in result['factors']
