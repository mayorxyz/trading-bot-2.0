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


if __name__ == "__main__":
    test_swing_high_confirmed_after_right_n()
    test_swing_low_confirmed_after_right_n()
    test_no_lookahead_with_real_data()
    test_edge_case_start_of_dataframe()
    test_calculate_market_structure_wrapper()
    print("\n✅ All Stage 1 tests passed!")
