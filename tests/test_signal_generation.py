import pytest
from datetime import datetime, timedelta
import polars as pl

from tbb.signals.generator import SignalGenerator


def test_entry_sl_tp_math_and_structural_stop():
    """Verify Entry=CE, SL beyond structural+buffer, TP1=1.0R, TP2=2.5R."""
    data = {
        "close": [100.0],
        "is_bullish_fvg_active": [True],
        "bullish_fvg_ce": [100.0],
        "swing_low_price": [95.0],
        "is_low_sweep": [False],
        "atr_14": [2.0],
        "is_high_sweep": [False],
        "is_bullish_bos": [True],
        "is_bearish_bos": [False],
        "bullish_ob_top": [None],
        "bullish_ob_bottom": [None],
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.55, "factors": ["BOS", "Retracement"]}
    
    gen = SignalGenerator({"timeframe": "15m"})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is not None
    sig = result
    
    assert sig.entry_price == 100.0
    assert sig.stop_loss == 94.8
    assert sig.take_profit_1 == 105.2
    assert sig.take_profit_2 == 113.0
    assert sig.stop_loss < 95.0


def test_signal_rejected_below_threshold():
    """Signal not generated when confluence score too low."""
    data = {
        "close": [100.0],
        "is_bullish_fvg_active": [False],
        "bullish_fvg_ce": [None],
        "swing_low_price": [None],
        "is_low_sweep": [False],
        "atr_14": [2.0],
        "is_high_sweep": [False],
        "is_bullish_bos": [False],
        "is_bearish_bos": [False],
        "bullish_ob_top": [None],
        "bullish_ob_bottom": [None],
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.30, "factors": ["Weak"]}
    
    gen = SignalGenerator({"timeframe": "15m"})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is None


def test_expiry_by_timeframe():
    """Dynamic expiry based on timeframe."""
    data = {
        "close": [100.0],
        "is_bullish_fvg_active": [True],
        "bullish_fvg_ce": [100.0],
        "swing_low_price": [95.0],
        "is_low_sweep": [False],
        "atr_14": [2.0],
        "is_high_sweep": [False],
        "is_bullish_bos": [True],
        "is_bearish_bos": [False],
        "bullish_ob_top": [None],
        "bullish_ob_bottom": [None],
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.55, "factors": ["BOS"]}
    ts = datetime.now()
    
    gen_5m = SignalGenerator({"timeframe": "5m"})
    sig_5m = gen_5m.generate_signal(df, confluence, "BTCUSDT", ts)
    assert sig_5m.expiry - ts == timedelta(hours=4)
    
    gen_15m = SignalGenerator({"timeframe": "15m"})
    sig_15m = gen_15m.generate_signal(df, confluence, "BTCUSDT", ts)
    assert sig_15m.expiry - ts == timedelta(hours=24)
    
    gen_4h = SignalGenerator({"timeframe": "4h"})
    sig_4h = gen_4h.generate_signal(df, confluence, "BTCUSDT", ts)
    assert sig_4h.expiry - ts == timedelta(hours=48)


def test_no_valid_entry_zone():
    """Signal rejected when no FVG or OB present."""
    data = {
        "close": [100.0],
        "is_bullish_fvg_active": [False],
        "bullish_fvg_ce": [None],
        "swing_low_price": [95.0],
        "is_low_sweep": [False],
        "atr_14": [2.0],
        "is_high_sweep": [True],
        "is_bullish_bos": [False],
        "is_bearish_bos": [False],
        "bullish_ob_top": [None],
        "bullish_ob_bottom": [None],
        "bearish_ob_top": [None],
        "bearish_ob_bottom": [None],
    }
    df = pl.DataFrame(data)
    confluence = {"score": 0.55, "factors": ["Sweep"]}
    
    gen = SignalGenerator({"timeframe": "15m"})
    result = gen.generate_signal(df, confluence, "BTCUSDT", datetime.now())
    
    assert result is None
