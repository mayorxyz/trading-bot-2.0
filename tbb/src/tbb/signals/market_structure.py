"""
Market Structure Detection using Polars.
Implements Smart Money Concepts (SMC) patterns with ZERO look-ahead bias.

All detection uses only historical data up to the current bar.
Signals are confirmed only after the required number of future bars have closed.
"""

import polars as pl
from tbb.core.config import settings


def detect_swings(df: pl.DataFrame, strength: int = None) -> pl.DataFrame:
    """
    Detect swing highs and lows using a fractal/pivot method.
    
    A swing high at bar i is confirmed only at bar i + strength (zero look-ahead).
    Uses Polars rolling_max/rolling_min for vectorized performance.
    
    Parameters
    ----------
    df : pl.DataFrame
        OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
    strength : int, optional
        Number of bars on each side to confirm pivot (default from config)
        strength=2 means 5-bar fractal (2 left, center, 2 right)
        strength=1 means 3-bar fractal (1 left, center, 1 right)
    
    Returns
    -------
    pl.DataFrame
        Original DataFrame with added columns:
        - swing_high: Boolean (True when confirmed)
        - swing_low: Boolean (True when confirmed)
        - swing_high_price: Float (price level, populated only on confirmation)
        - swing_low_price: Float (price level, populated only on confirmation)
    """
    # Get strength from config if not provided
    if strength is None:
        strength = getattr(settings, 'FRACTAL_STRENGTH', 2)
    
    window_size = 2 * strength + 1
    
    # Detect potential swing highs (center bar is highest in window)
    # This uses a centered window, so we need to shift the result to enforce look-ahead
    df = df.with_columns([
        # Rolling max over the window
        pl.col("high").rolling_max(window_size=window_size).alias("rolling_high"),
        # Check if current high equals the rolling max AND is unique (strictly greater)
        (pl.col("high") == pl.col("rolling_high")).alias("is_candidate_high"),
    ])
    
    # To ensure it's strictly greater than neighbors, we check against shifted values
    # This is a simplified approach; for strict fractal we'd check each neighbor
    df = df.with_columns([
        # More robust: check if high is greater than all bars in left and right windows
        (pl.col("high") > pl.col("high").shift(1).rolling_max(window_size=strength)).alias("left_check_high"),
        (pl.col("high") > pl.col("high").shift(-strength).rolling_min(window_size=strength)).alias("right_check_high_temp"),
    ])
    
    # Actually, let's use a cleaner approach with explicit slicing via shift
    # Swing High: high[i] > high[i-strength:i] AND high[i] > high[i+1:i+strength]
    # Since we can't see future, we detect the PATTERN then shift the confirmation
    
    # Simpler robust method:
    # 1. Find bars where high is local max over 2*strength+1 window
    # 2. Shift the boolean result by 'strength' bars to confirm only after right side closes
    
    df = df.with_columns([
        # Check if current high is the max of the LOOKBACK window (left side + current)
        pl.col("high").rolling_max(window_size=strength + 1).alias("max_left"),
        # Identify candidate: current high equals max of left window
        (pl.col("high") == pl.col("max_left")).alias("candidate_high"),
    ])
    
    # Now we need to wait 'strength' bars to confirm the right side
    # We'll mark the candidate, then shift the confirmation flag
    
    # Better approach: use shift to compare with future bars once they arrive
    # For each bar, we check if it remains the high for the next 'strength' bars
    
    # Most reliable vectorized approach:
    # Create a boolean where high[i] is strictly greater than high[i-strength:i] and high[i+1:i+strength]
    # Then shift this boolean by 'strength' to get confirmation
    
    # Left side check: high[i] > max(high[i-strength:i-1])
    left_max = pl.col("high").shift(1).rolling_max(window_size=strength)
    
    # Right side check will be done by shifting the final result
    
    df = df.with_columns([
        # Candidate: high is greater than previous 'strength' bars
        (pl.col("high") > left_max).alias("passed_left_high"),
        # Also check it's greater than the next 'strength' bars (this creates a future-looking flag)
        # We'll compute this, then shift it forward by 'strength' to make it confirmation-only
        (pl.col("high") > pl.col("high").shift(-1).rolling_min(window_size=strength)).alias("passed_right_high_raw"),
    ])
    
    # Combine: passed both left and right checks
    df = df.with_columns([
        (pl.col("passed_left_high") & pl.col("passed_right_high_raw")).alias("swing_high_raw"),
    ])
    
    # CRITICAL: Shift by 'strength' to enforce zero look-ahead
    # The signal at index i is only usable at index i + strength
    df = df.with_columns([
        pl.col("swing_high_raw").shift(strength).alias("swing_high"),
        # Populate price only on confirmed swing
        pl.when(pl.col("swing_high"))
          .then(pl.col("high").shift(strength))
          .otherwise(None)
          .alias("swing_high_price"),
    ])
    
    # === SWING LOWS (mirror logic) ===
    left_min = pl.col("low").shift(1).rolling_min(window_size=strength)
    
    df = df.with_columns([
        (pl.col("low") < left_min).alias("passed_left_low"),
        (pl.col("low") < pl.col("low").shift(-1).rolling_max(window_size=strength)).alias("passed_right_low_raw"),
        (pl.col("passed_left_low") & pl.col("passed_right_low_raw")).alias("swing_low_raw"),
        pl.col("swing_low_raw").shift(strength).alias("swing_low"),
        pl.when(pl.col("swing_low"))
          .then(pl.col("low").shift(strength))
          .otherwise(None)
          .alias("swing_low_price"),
    ])
    
    # Clean up temporary columns
    drop_cols = [
        "rolling_high", "is_candidate_high", "left_check_high", "right_check_high_temp",
        "max_left", "candidate_high", "passed_left_high", "passed_right_high_raw", 
        "swing_high_raw", "passed_left_low", "passed_right_low_raw", "swing_low_raw"
    ]
    
    # Only drop if they exist (some might have been intermediate steps)
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(existing_drops)
    
    return df


def calculate_market_structure(df: pl.DataFrame, config: dict = None) -> pl.DataFrame:
    """
    Main entry point for market structure calculation.
    Currently wraps swing detection; will be extended in Stage 2.
    
    Parameters
    ----------
    df : pl.DataFrame
        OHLCV DataFrame
    config : dict, optional
        Configuration overrides (not used yet, reserved for future stages)
    
    Returns
    -------
    pl.DataFrame
        DataFrame with market structure columns appended
    """
    if config is None:
        config = {}
    
    strength = config.get('fractal_strength', getattr(settings, 'FRACTAL_STRENGTH', 2))
    
    df = detect_swings(df, strength=strength)
    
    # TODO: Stage 2 will add FVG, BOS, Sweep, OB detection here
    
    return df
