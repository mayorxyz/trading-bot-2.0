"""
Tests for Market Structure Detection (Stage 1: Swing High/Low).
Focus: Zero look-ahead bias verification.
"""

import polars as pl
from datetime import datetime, timedelta
from tbb.signals.market_structure import detect_swings, calculate_market_structure


def create_test_dataframe() -> pl.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    n = 50
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    data = {
        'timestamp': timestamps,
        'open': [100.0 + i * 0.5 for i in range(n)],
        'high': [102.0 + i * 0.5 if i % 10 != 5 else 110.0 + i * 0.5 for i in range(n)],  # Peak at index 5
        'low': [98.0 + i * 0.5 for i in range(n)],
        'close': [101.0 + i * 0.5 for i in range(n)],
        'volume': [1000.0] * n,
    }
    
    return pl.DataFrame(data)


def test_swing_high_confirmed_after_right_n():
    """
    CRITICAL LOOK-AHEAD TEST:
    Verify a swing high at index i is NOT confirmed until index i + strength.
    With strength=2, a peak at index 10 should only show swing_high=True at index 12.
    """
    df = create_test_dataframe()
    
    # Create a clear swing high at index 10
    # Make index 10 the highest among indices 8-12
    df = df.with_columns([
        pl.when(pl.col('timestamp') == df['timestamp'][10])
          .then(150.0)  # Clear peak
          .otherwise(pl.col('high'))
          .alias('high')
    ])
    
    result = detect_swings(df, strength=2)
    
    # Index 10: swing_high should be NULL (not yet confirmed)
    assert result['swing_high'][10] is None or result['swing_high'][10] == False, \
        f"Look-ahead bug: swing_high confirmed at index 10, should be NULL"
    
    # Index 11: swing_high should still be NULL (only 1 bar after peak)
    assert result['swing_high'][11] is None or result['swing_high'][11] == False, \
        f"Look-ahead bug: swing_high confirmed at index 11, should be NULL"
    
    # Index 12: swing_high should be TRUE (2 bars after peak = confirmed)
    assert result['swing_high'][12] is True, \
        f"Swing high not confirmed at index 12 (expected True, got {result['swing_high'][12]})"
    
    # Verify price is populated correctly
    assert result['swing_high_price'][12] == 150.0, \
        f"Swing high price incorrect: expected 150.0, got {result['swing_high_price'][12]}"
    
    print("✓ test_swing_high_confirmed_after_right_n PASSED")


def test_swing_low_confirmed_after_right_n():
    """
    Mirror test for swing lows.
    With strength=2, a trough at index 10 should only show swing_low=True at index 12.
    """
    df = create_test_dataframe()
    
    # Create a clear swing low at index 10
    df = df.with_columns([
        pl.when(pl.col('timestamp') == df['timestamp'][10])
          .then(50.0)  # Clear trough
          .otherwise(pl.col('low'))
          .alias('low')
    ])
    
    result = detect_swings(df, strength=2)
    
    # Index 10, 11: should be NULL
    assert result['swing_low'][10] is None or result['swing_low'][10] == False
    assert result['swing_low'][11] is None or result['swing_low'][11] == False
    
    # Index 12: should be TRUE
    assert result['swing_low'][12] is True, \
        f"Swing low not confirmed at index 12"
    
    assert result['swing_low_price'][12] == 50.0
    
    print("✓ test_swing_low_confirmed_after_right_n PASSED")


def test_no_lookahead_with_real_data():
    """
    Test with a more realistic price pattern to verify no look-ahead bias.
    Creates multiple swings and verifies each is only confirmed after the required lag.
    """
    n = 100
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    # Create a zig-zag pattern with clear swings every 10 bars
    highs = []
    lows = []
    for i in range(n):
        if i % 20 == 10:  # Peak
            highs.append(200.0)
            lows.append(100.0 + (i // 20) * 5)
        elif i % 20 == 0:  # Trough
            highs.append(120.0 + (i // 20) * 5)
            lows.append(80.0)
        else:  # Normal
            highs.append(150.0 + (i // 20) * 2)
            lows.append(100.0 + (i // 20) * 2)
    
    data = {
        'timestamp': timestamps,
        'open': [100.0 + i * 0.1 for i in range(n)],
        'high': highs,
        'low': lows,
        'close': [125.0 + i * 0.1 for i in range(n)],
        'volume': [1000.0] * n,
    }
    
    df = pl.DataFrame(data)
    result = detect_swings(df, strength=2)
    
    # Verify first peak at index 10 is confirmed at index 12
    assert result['swing_high'][12] is True
    assert result['swing_high'][10] is None or result['swing_high'][10] == False
    assert result['swing_high'][11] is None or result['swing_high'][11] == False
    
    # Verify first trough at index 0 is confirmed at index 2
    assert result['swing_low'][2] is True
    assert result['swing_low'][0] is None or result['swing_low'][0] == False
    assert result['swing_low'][1] is None or result['swing_low'][1] == False
    
    print("✓ test_no_lookahead_with_real_data PASSED")


def test_edge_case_start_of_dataframe():
    """
    Verify graceful handling of start-of-dataframe where shifts produce nulls.
    """
    df = create_test_dataframe()
    result = detect_swings(df, strength=2)
    
    # First few rows should have nulls due to shift/rolling window
    assert result['swing_high'][0] is None
    assert result['swing_high'][1] is None
    
    print("✓ test_edge_case_start_of_dataframe PASSED")


def test_calculate_market_structure_wrapper():
    """
    Test the main entry point wrapper function.
    """
    df = create_test_dataframe()
    config = {'fractal_strength': 2}
    
    result = calculate_market_structure(df, config)
    
    # Should have all swing columns
    assert 'swing_high' in result.columns
    assert 'swing_low' in result.columns
    assert 'swing_high_price' in result.columns
    assert 'swing_low_price' in result.columns
    
    print("✓ test_calculate_market_structure_wrapper PASSED")


def test_fvg_displacement_filter():
    """Assert that a gap smaller than 0.5*ATR is filtered out."""
    n = 20
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    # Create data with ATR ~10.0, so 0.5*ATR = 5.0
    # Make a small FVG gap of 3.0 (should be rejected)
    data = {
        'timestamp': timestamps,
        'open': [100.0] * n,
        'high': [105.0] * n,
        'low': [95.0] * n,
        'close': [100.0] * n,
        'volume': [1000.0] * n,
    }
    df = pl.DataFrame(data)
    
    # Candle 2: create bullish FVG (Low[2] > High[0])
    # Gap = 97.0 - 95.0 = 2.0 (less than 5.0 threshold)
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[2]).then(97.0).otherwise(pl.col('low')).alias('low'),
        pl.when(pl.col('timestamp') == timestamps[0]).then(95.0).otherwise(pl.col('high')).alias('high'),
    ])
    
    result = calculate_market_structure(df, {})
    
    # Should be False because gap < 0.5 * ATR
    assert result['is_bullish_fvg_active'][2] == False, \
        f"FVG should be filtered out (gap too small), got {result['is_bullish_fvg_active'][2]}"
    
    print("✓ test_fvg_displacement_filter PASSED")


def test_sweep_wick_ratio():
    """Assert weak wick (< 2x body) is not marked as sweep."""
    n = 20
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    # Create swing low at index 5
    # Create potential sweep at index 10
    data = {
        'timestamp': timestamps,
        'open': [100.0] * n,
        'high': [110.0] * n,
        'low': [90.0 if i == 5 else 85.0 if i == 10 else 95.0 for i in range(n)],
        'close': [100.0 if i != 10 else 98.0 for i in range(n)],  # Close back inside at 10
        'volume': [1000.0] * n,
    }
    df = pl.DataFrame(data)
    
    # At index 10: Low=85 (below swing low 90), Close=98 (above 90)
    # But make wick small (only 1.5x body) to fail sweep filter
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[10]).then(96.0).otherwise(pl.col('open')).alias('open'),
        pl.when(pl.col('timestamp') == timestamps[10]).then(98.0).otherwise(pl.col('close')).alias('close'),
        # Wick = 90-85=5, Body = |98-96|=2, Ratio = 2.5 (passes)
        # Let's make it fail: Wick = 2, Body = 2, Ratio = 1.0 (fails)
        pl.when(pl.col('timestamp') == timestamps[10]).then(89.0).otherwise(pl.col('low')).alias('low'),
    ])
    
    result = calculate_market_structure(df, {'sweep_wick_ratio': 2.0})
    
    # Should be False because wick/body ratio < 2.0
    assert result['is_low_sweep'][10] == False, \
        f"Sweep should be filtered out (wick ratio too small), got {result['is_low_sweep'][10]}"
    
    print("✓ test_sweep_wick_ratio PASSED")


def test_bos_volume_confirmation():
    """Assert BOS requires Volume > threshold OR FVG on candle."""
    n = 20
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    # Create swing high at index 5 (High=110)
    # Create BOS candidate at index 10 (Close > 110)
    # But with LOW volume (no FVG either) -> should NOT be BOS
    data = {
        'timestamp': timestamps,
        'open': [100.0] * n,
        'high': [110.0 if i == 5 else 105.0 for i in range(n)],
        'low': [95.0] * n,
        'close': [100.0 if i != 10 else 115.0 for i in range(n)],  # Close above swing at 10
        'volume': [1000.0 if i != 10 else 500.0 for i in range(n)],  # Low volume at 10
    }
    df = pl.DataFrame(data)
    
    result = calculate_market_structure(df, {'bos_volume_mult': 1.5})
    
    # Should be False because low volume AND no FVG
    assert result['is_bullish_bos'][10] == False, \
        f"BOS should require volume or FVG, got {result['is_bullish_bos'][10]}"
    
    print("✓ test_bos_volume_confirmation PASSED")


def test_ob_with_active_fvg():
    """Assert OB only identified when BOS has active FVG."""
    n = 30
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    # Bearish candle at index 10 (potential OB)
    # Bullish BOS at index 20 with FVG
    data = {
        'timestamp': timestamps,
        'open': [100.0] * n,
        'high': [105.0 if i != 10 else 110.0 for i in range(n)],
        'low': [95.0] * n,
        'close': [100.0 if i != 10 else 90.0 for i in range(n)],  # Bearish at 10
        'volume': [2000.0] * n,  # High volume for BOS
    }
    df = pl.DataFrame(data)
    
    # Create FVG at index 20 (Bullish: Low[20] > High[18])
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[20]).then(108.0).otherwise(pl.col('low')).alias('low'),
        pl.when(pl.col('timestamp') == timestamps[18]).then(105.0).otherwise(pl.col('high')).alias('high'),
        # Make index 20 close above swing high
        pl.when(pl.col('timestamp') == timestamps[20]).then(115.0).otherwise(pl.col('close')).alias('close'),
    ])
    
    result = calculate_market_structure(df, {'ob_scan_back': 15})
    
    # OB should be found at index 20 (pointing to candle 10)
    ob_top = result['bullish_ob_top'][20]
    ob_bot = result['bullish_ob_bottom'][20]
    
    assert ob_top is not None, "OB top should be populated"
    assert ob_bot is not None, "OB bottom should be populated"
    # OB should match candle 10's high/low
    assert ob_top == 110.0, f"OB top should be 110.0, got {ob_top}"
    assert ob_bot == 90.0, f"OB bottom should be 90.0, got {ob_bot}"
    
    print("✓ test_ob_with_active_fvg PASSED")


def test_ob_index_alignment():
    """
    CRITICAL: Verify OB scanning uses correct original indices.
    Setup: OB anchor at row 10, BOS at row 15.
    Assert: OB values at row 15 match row 10's high/low.
    """
    n = 30
    timestamps = [datetime(2024, 1, 1, 0, i) for i in range(n)]
    
    data = {
        'timestamp': timestamps,
        'open': [100.0] * n,
        'high': [105.0] * n,
        'low': [95.0] * n,
        'close': [100.0] * n,
        'volume': [1000.0] * n,
    }
    df = pl.DataFrame(data)
    
    # Row 10: Bearish candle (OB candidate)
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[10]).then(90.0).otherwise(pl.col('open')).alias('open'),
        pl.when(pl.col('timestamp') == timestamps[10]).then(80.0).otherwise(pl.col('close')).alias('close'),
        pl.when(pl.col('timestamp') == timestamps[10]).then(95.0).otherwise(pl.col('high')).alias('high'),
        pl.when(pl.col('timestamp') == timestamps[10]).then(75.0).otherwise(pl.col('low')).alias('low'),
    ])
    
    # Row 15: Bullish BOS with FVG
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[15]).then(True).otherwise(pl.col('__placeholder__')).alias('is_bullish_bos_temp'),
        pl.when(pl.col('timestamp') == timestamps[15]).then(True).otherwise(pl.col('__placeholder__')).alias('is_bullish_fvg_temp'),
        pl.when(pl.col('timestamp') == timestamps[15]).then(110.0).otherwise(pl.col('open')).alias('open'),
        pl.when(pl.col('timestamp') == timestamps[15]).then(130.0).otherwise(pl.col('close')).alias('close'),
        pl.when(pl.col('timestamp') == timestamps[15]).then(2000.0).otherwise(pl.col('volume')).alias('volume'),
    ])
    
    # Manually set BOS and FVG flags (since calculate_market_structure computes them)
    # We need the natural flow, so let's just ensure the setup triggers BOS+FVG
    # Create FVG: Low[15] > High[13]
    df = df.with_columns([
        pl.when(pl.col('timestamp') == timestamps[15]).then(108.0).otherwise(pl.col('low')).alias('low'),
        pl.when(pl.col('timestamp') == timestamps[13]).then(105.0).otherwise(pl.col('high')).alias('high'),
    ])
    
    result = calculate_market_structure(df, {'ob_scan_back': 10})
    
    # Check OB at row 15
    ob_top = result['bullish_ob_top'][15]
    ob_bot = result['bullish_ob_bottom'][15]
    
    # Should match row 10: High=95.0, Low=75.0
    assert ob_top == 95.0, f"OB top should be 95.0 (row 10 high), got {ob_top}"
    assert ob_bot == 75.0, f"OB bottom should be 75.0 (row 10 low), got {ob_bot}"
    
    # Verify rows without BOS don't have OB
    assert result['bullish_ob_top'][14] is None
    assert result['bullish_ob_top'][16] is None
    
    print("✓ test_ob_index_alignment PASSED")


if __name__ == "__main__":
    test_swing_high_confirmed_after_right_n()
    test_swing_low_confirmed_after_right_n()
    test_no_lookahead_with_real_data()
    test_edge_case_start_of_dataframe()
    test_calculate_market_structure_wrapper()
    test_fvg_displacement_filter()
    test_sweep_wick_ratio()
    test_bos_volume_confirmation()
    test_ob_with_active_fvg()
    test_ob_index_alignment()
    print("\n✅ All Stage 1 & 2 tests passed!")
