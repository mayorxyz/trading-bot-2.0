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
    Master pipeline: Swings -> FVG -> BOS/Sweep -> OB
    All parameters read from config dict or fallback to settings.
    
    Parameters
    ----------
    df : pl.DataFrame
        OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
    config : dict, optional
        Configuration overrides for market structure parameters
        
    Returns
    -------
    pl.DataFrame
        DataFrame with all market structure columns appended:
        - swing_high, swing_low, swing_high_price, swing_low_price
        - is_bullish_fvg_active, bullish_fvg_ce, is_bearish_fvg_active, bearish_fvg_ce
        - is_bullish_bos, is_bearish_bos, is_high_sweep, is_low_sweep
        - bullish_ob_top, bullish_ob_bottom, bearish_ob_top, bearish_ob_bottom
    """
    if config is None:
        config = {}
    
    # Get Config Values
    fractal_strength = config.get('fractal_strength', getattr(settings, 'FRACTAL_STRENGTH', 2))
    fvg_gap_mult = config.get('fvg_gap_atr_mult', getattr(settings, 'FVG_GAP_ATR_MULT', 0.5))
    fvg_body_mult = config.get('fvg_body_atr_mult', getattr(settings, 'FVG_BODY_ATR_MULT', 0.5))
    sweep_ratio = config.get('sweep_wick_ratio', getattr(settings, 'SWEEP_WICK_RATIO', 2.0))
    bos_vol_mult = config.get('bos_volume_mult', getattr(settings, 'BOS_VOLUME_MULT', 1.5))
    ob_scan_back = config.get('ob_scan_back', getattr(settings, 'OB_SCAN_BACK', 25))

    # 1. SWING DETECTION (Stage 1)
    df = detect_swings(df, strength=fractal_strength)
    
    # 2. ATR Calculation (needed for FVG displacement filter)
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs()
    )
    df = df.with_columns(tr.rolling_mean(14).alias("atr_14"))
    
    # 3. FVG DETECTION
    # Bullish FVG: Low[i] > High[i-2]
    # Bearish FVG: High[i] < Low[i-2]
    df = df.with_columns([
        (pl.col("low") > pl.col("high").shift(2)).alias("raw_bull_fvg"),
        (pl.col("high") < pl.col("low").shift(2)).alias("raw_bear_fvg")
    ])
    
    # Displacement Filter: Gap Size AND Body Size must exceed thresholds
    # Gap size for bullish: Low[i] - High[i-2]
    # Body size: |Close[i-1] - Open[i-1]| (middle candle of the 3-candle pattern)
    gap_size_bull = pl.col("low") - pl.col("high").shift(2)
    gap_size_bear = pl.col("high").shift(2) - pl.col("low")
    body_size = (pl.col("close").shift(1) - pl.col("open").shift(1)).abs()
    
    df = df.with_columns([
        # Bullish FVG with displacement filter
        (
            pl.col("raw_bull_fvg") & 
            (gap_size_bull >= (fvg_gap_mult * pl.col("atr_14"))) & 
            (body_size >= (fvg_body_mult * pl.col("atr_14")))
        ).alias("is_bullish_fvg_active"),
        
        # Bearish FVG with displacement filter
        (
            pl.col("raw_bear_fvg") & 
            (gap_size_bear >= (fvg_gap_mult * pl.col("atr_14"))) & 
            (body_size >= (fvg_body_mult * pl.col("atr_14")))
        ).alias("is_bearish_fvg_active")
    ])
    
    # Calculate CE (Consequent Encroachment - 50% midpoint)
    df = df.with_columns([
        pl.when(pl.col("is_bullish_fvg_active"))
          .then((pl.col("low") + pl.col("high").shift(2)) / 2)
          .otherwise(None).cast(pl.Float64).alias("bullish_fvg_ce"),
          
        pl.when(pl.col("is_bearish_fvg_active"))
          .then((pl.col("high").shift(2) + pl.col("low")) / 2)
          .otherwise(None).cast(pl.Float64).alias("bearish_fvg_ce")
    ])
    
    # Mitigation tracking via forward-fill state
    # An FVG stays active until price trades through the CE
    # We'll use a cumulative approach: mark "breach" events, then cumulative max to deactivate
    
    # For bullish: deactivated when Low <= CE
    df = df.with_columns(
        (pl.col("low") <= pl.col("bullish_fvg_ce")).alias("bullish_fvg_breach")
    )
    
    # Forward-fill active status until breach occurs
    # Use cum_sum of breaches to create groups, then keep only first in group
    df = df.with_columns(
        pl.col("bullish_fvg_breach").fill_null(0).cum_sum().alias("bull_breach_cumsum")
    )
    
    # Reset active status after breach
    df = df.with_columns(
        pl.when(pl.col("bull_breach_cumsum") > 0)
          .then(False)
          .otherwise(pl.col("is_bullish_fvg_active"))
          .alias("is_bullish_fvg_active")
    )
    
    # Same for bearish
    df = df.with_columns(
        (pl.col("high") >= pl.col("bearish_fvg_ce")).alias("bearish_fvg_breach")
    )
    df = df.with_columns(
        pl.col("bearish_fvg_breach").fill_null(0).cum_sum().alias("bear_breach_cumsum")
    )
    df = df.with_columns(
        pl.when(pl.col("bear_breach_cumsum") > 0)
          .then(False)
          .otherwise(pl.col("is_bearish_fvg_active"))
          .alias("is_bearish_fvg_active")
    )
    
    # Clean up temp breach columns
    df = df.drop(["raw_bull_fvg", "raw_bear_fvg", "bullish_fvg_breach", "bull_breach_cumsum", 
                  "bearish_fvg_breach", "bear_breach_cumsum"])

    # 4. BOS & SWEEP DETECTION
    # Forward-fill the last confirmed swing price
    df = df.with_columns([
        pl.col("swing_high_price").fill_null(strategy="forward").alias("last_swing_high"),
        pl.col("swing_low_price").fill_null(strategy="forward").alias("last_swing_low")
    ])
    
    # Volume MA for BOS filter
    df = df.with_columns(
        pl.col("volume").rolling_mean(20).alias("vol_ma_20")
    )
    
    # Sweep Detection: Wick beyond swing, Close back inside
    # Long Sweep (Low Sweep): Low < last_swing_low, Close > last_swing_low
    # Wick-to-body ratio filter: wick >= sweep_ratio * body
    
    lower_wick = pl.col("open").min(pl.col("close")) - pl.col("low")
    upper_wick = pl.col("high") - pl.col("open").max(pl.col("close"))
    body = (pl.col("close") - pl.col("open")).abs()
    
    df = df.with_columns([
        # Low Sweep
        (
            (pl.col("low") < pl.col("last_swing_low")) & 
            (pl.col("close") > pl.col("last_swing_low")) &
            (lower_wick >= (sweep_ratio * body))
        ).alias("is_low_sweep"),
        
        # High Sweep
        (
            (pl.col("high") > pl.col("last_swing_high")) & 
            (pl.col("close") < pl.col("last_swing_high")) &
            (upper_wick >= (sweep_ratio * body))
        ).alias("is_high_sweep")
    ])
    
    # BOS Detection: Close beyond swing
    # Requires: Volume > threshold OR FVG created on BOS candle
    df = df.with_columns([
        # Bullish BOS
        (
            (pl.col("close") > pl.col("last_swing_high")) & 
            ((pl.col("volume") > (bos_vol_mult * pl.col("vol_ma_20"))) | pl.col("is_bullish_fvg_active"))
        ).alias("is_bullish_bos"),
        
        # Bearish BOS
        (
            (pl.col("close") < pl.col("last_swing_low")) & 
            ((pl.col("volume") > (bos_vol_mult * pl.col("vol_ma_20"))) | pl.col("is_bearish_fvg_active"))
        ).alias("is_bearish_bos")
    ])

    # 5. ORDER BLOCK DETECTION (Polars Native with Row Index)
    # Add index to preserve original positions (fixes indexing bug)
    df_with_idx = df.with_row_index("original_idx")
    
    # Initialize OB columns as lists
    ob_bull_top = [None] * len(df)
    ob_bull_bot = [None] * len(df)
    ob_bear_top = [None] * len(df)
    ob_bear_bot = [None] * len(df)
    
    # Filter for BOS events only (sparse iteration for performance)
    bos_rows = df_with_idx.filter(
        pl.col("is_bullish_bos") | pl.col("is_bearish_bos")
    ).to_dicts()
    
    for row in bos_rows:
        idx = row["original_idx"]
        if idx < ob_scan_back:
            continue  # Not enough history to scan back
            
        start_scan = idx - ob_scan_back
        window_len = idx - start_scan
        
        # Slice window using Polars
        window = df.slice(start_scan, window_len)
        
        if row["is_bullish_bos"] and row["is_bullish_fvg_active"]:
            # Find last bearish candle (close < open) in window
            bearish = window.filter(pl.col("close") < pl.col("open"))
            if len(bearish) > 0:
                last_candle = bearish.tail(1)
                ob_bull_top[idx] = last_candle["high"][0]
                ob_bull_bot[idx] = last_candle["low"][0]
                
        elif row["is_bearish_bos"] and row["is_bearish_fvg_active"]:
            # Find last bullish candle (close > open) in window
            bullish = window.filter(pl.col("close") > pl.col("open"))
            if len(bullish) > 0:
                last_candle = bullish.tail(1)
                ob_bear_top[idx] = last_candle["high"][0]
                ob_bear_bot[idx] = last_candle["low"][0]

    # Add OB columns to dataframe
    df = df.with_columns([
        pl.Series(ob_bull_top, dtype=pl.Float64).alias("bullish_ob_top"),
        pl.Series(ob_bull_bot, dtype=pl.Float64).alias("bullish_ob_bottom"),
        pl.Series(ob_bear_top, dtype=pl.Float64).alias("bearish_ob_top"),
        pl.Series(ob_bear_bot, dtype=pl.Float64).alias("bearish_ob_bottom")
    ])
    
    # Cleanup temporary columns
    cols_to_drop = ["atr_14", "vol_ma_20", "last_swing_high", "last_swing_low", 
                    "original_idx"]
    existing_cols = df.columns
    drop_list = [c for c in cols_to_drop if c in existing_cols]
    
    df = df.drop(drop_list)
    
    return df
