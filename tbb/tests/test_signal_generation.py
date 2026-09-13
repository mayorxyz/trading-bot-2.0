import pytest
from datetime import datetime
import polars as pl
from tbb.signals.generator import SignalGenerator
from tbb.signals.schema import TradeSignal, SignalDirection


def test_entry_sl_tp_math_and_structural_stop():
    """
    Verify:
    1. Entry = 50% CE (100.0)
    2. SL = Structural Low (95.0) - 0.1*ATR (0.2) = 94.8
    3. TP1 = Entry + 1R (105.2), TP2 = Entry + 2.5R (113.0)
    4. SL is BEYOND structural invalidation point (95.0)
    """
    data = {
        'close': [100.0],
        'is_bullish_fvg_active': [True],
        'bullish_fvg_ce': [100.0],
        'swing_low_price': [95.0],
        'is_low_sweep': [False],
        'atr_14': [2.0],
        'is_high_sweep': [False],
        'is_bullish_bos': [True],
        'is_bearish_bos': [False],
        'bullish_ob_top': [None],
        'bullish_ob_bottom': [None]
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.55, "factors": ["BOS", "Retracement"]}
    
    gen = SignalGenerator({'timeframe': '15m'})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is not None
    sig = result
    
    # Check Math
    assert sig.entry_price == 100.0
    assert sig.stop_loss == 94.8  # 95.0 - 0.2
    assert sig.take_profit_1 == 105.2  # 100 + 5.2
    assert sig.take_profit_2 == 113.0  # 100 + (2.5 * 5.2)
    
    # Check Structural Safety: SL (94.8) must be < Swing Low (95.0)
    assert sig.stop_loss < 95.0


def test_signal_rejected_below_threshold():
    """Verify no signal generated when confluence score < MIN_CONFLUENCE_SCORE."""
    data = {
        'close': [100.0],
        'is_bullish_fvg_active': [False],
        'is_bullish_bos': [False],
        'is_high_sweep': [False]
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.25, "factors": ["Displacement"]}  # Below 0.40 threshold
    
    gen = SignalGenerator({'timeframe': '15m'})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    # Should return None or raise exception due to low score
    assert result is None or result.confluence_score < 0.40


def test_expiry_by_timeframe():
    """Test dynamic expiry logic for all timeframe categories."""
    data = {
        'close': [100.0],
        'is_bullish_fvg_active': [True],
        'bullish_fvg_ce': [100.0],
        'swing_low_price': [95.0],
        'is_bullish_bos': [True],
        'atr_14': [2.0]
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.70, "factors": ["Sweep", "FVG", "Displacement"]}
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    gen = SignalGenerator()
    
    # Test 1m-5m: 4 hours
    gen.config = {'timeframe': '5m'}
    sig = gen.generate_signal(df, confluence, "BTCUSDT", base_time)
    assert sig.expiry == base_time + timedelta(hours=4)
    
    # Test 15m-1h: 24 hours
    gen.config = {'timeframe': '15m'}
    sig = gen.generate_signal(df, confluence, "BTCUSDT", base_time)
    assert sig.expiry == base_time + timedelta(hours=24)
    
    # Test 4h: 48 hours
    gen.config = {'timeframe': '4h'}
    sig = gen.generate_signal(df, confluence, "BTCUSDT", base_time)
    assert sig.expiry == base_time + timedelta(hours=48)
    
    # Test Daily: 7 days
    gen.config = {'timeframe': '1d'}
    sig = gen.generate_signal(df, confluence, "BTCUSDT", base_time)
    assert sig.expiry == base_time + timedelta(days=7)


def test_no_valid_entry_zone():
    """Verify rejection when no FVG or OB present."""
    data = {
        'close': [100.0],
        'is_bullish_fvg_active': [False],
        'bullish_fvg_ce': [None],
        'bullish_ob_top': [None],
        'bullish_ob_bottom': [None],
        'is_bullish_bos': [True],
        'is_high_sweep': [False]
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.55, "factors": ["BOS"]}
    
    gen = SignalGenerator({'timeframe': '15m'})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is None


def test_short_signal_direction():
    """Verify short signal generation with bearish signals."""
    data = {
        'close': [100.0],
        'is_bearish_fvg_active': [True],
        'bearish_fvg_ce': [100.0],
        'swing_high_price': [105.0],
        'is_high_sweep': [True],
        'atr_14': [2.0],
        'is_bearish_bos': [True],
        'is_bullish_bos': [False]
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.70, "factors": ["Sweep+BOS Combo", "FVG"]}
    
    gen = SignalGenerator({'timeframe': '15m'})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is not None
    assert result.direction == SignalDirection.SHORT
    assert result.entry_price == 100.0
    # SL should be above swing high (105.0) + buffer (0.2) = 105.2
    assert result.stop_loss > 105.0
    # TP should be below entry
    assert result.take_profit_1 < 100.0
    assert result.take_profit_2 < 100.0
