"""Walk-Forward Validation Framework.

Implements rolling window analysis to detect overfitting:
- In-Sample (IS): 12 months for parameter optimization
- Out-of-Sample (OOS): 6 months for validation
- Roll forward by 3 months
- Calculate IS vs OOS decay to flag overfitting
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from tbb.core.logger import get_logger
from tbb.backtest.engine import BacktestEngine, BacktestConfig, BacktestMetrics

logger = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_metrics: Optional[BacktestMetrics] = None
    oos_metrics: Optional[BacktestMetrics] = None
    
    @property
    def is_months(self) -> int:
        """Get in-sample period in months."""
        return (self.is_end - self.is_start).days // 30
    
    @property
    def oos_months(self) -> int:
        """Get out-of-sample period in months."""
        return (self.oos_end - self.oos_start).days // 30
    
    @property
    def decay_ratio(self) -> float:
        """Calculate OOS Sharpe / IS Sharpe ratio."""
        if not self.is_metrics or not self.oos_metrics:
            return 1.0
        if self.is_metrics.sharpe_ratio == 0:
            return 1.0 if self.oos_metrics.sharpe_ratio == 0 else float('inf')
        return self.oos_metrics.sharpe_ratio / self.is_metrics.sharpe_ratio
    
    @property
    def is_overfitted(self) -> bool:
        """Check if this window shows severe overfitting."""
        return self.decay_ratio < 0.5


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""
    windows: list[WalkForwardWindow] = field(default_factory=list)
    stitched_oos_equity: list[tuple[datetime, float]] = field(default_factory=list)
    total_is_trades: int = 0
    total_oos_trades: int = 0
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_decay_ratio: float = 1.0
    overfitting_flagged: bool = False
    overfitting_windows: int = 0
    regime_labels: dict[str, str] = field(default_factory=dict)  # window_id -> regime
    
    @property
    def oos_expectancy(self) -> float:
        """Calculate overall OOS expectancy."""
        if not self.windows:
            return 0.0
        
        oos_pnls = []
        oos_counts = []
        for w in self.windows:
            if w.oos_metrics:
                oos_pnls.append(w.oos_metrics.total_pnl_r)
                oos_counts.append(w.oos_metrics.total_trades)
        
        if sum(oos_counts) == 0:
            return 0.0
        
        return sum(oos_pnls) / sum(oos_counts)
    
    @property
    def robustness_score(self) -> float:
        """Calculate overall robustness score (0-1).
        
        1.0 = perfect robustness (OOS >= IS)
        0.0 = complete overfitting (OOS = 0)
        """
        if not self.windows:
            return 0.0
        
        valid_windows = [w for w in self.windows if w.is_metrics and w.oos_metrics]
        if not valid_windows:
            return 0.0
        
        decay_ratios = [w.decay_ratio for w in valid_windows]
        # Clamp to 0-1 range
        clamped = [max(0, min(1, r)) for r in decay_ratios]
        return sum(clamped) / len(clamped)


class WalkForwardValidator:
    """Orchestrates walk-forward validation across multiple windows.
    
    Usage:
        validator = WalkForwardValidator(base_config)
        result = await validator.run()
        
        print(f"OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        print(f"Decay Ratio: {result.avg_decay_ratio:.2f}")
        print(f"Overfitting Flagged: {result.overfitting_flagged}")
    """
    
    def __init__(
        self,
        base_config: BacktestConfig,
        is_months: int = 12,
        oos_months: int = 6,
        roll_months: int = 3,
    ):
        """Initialize walk-forward validator.
        
        Args:
            base_config: Base backtest configuration
            is_months: In-sample period in months
            oos_months: Out-of-sample period in months
            roll_months: Roll forward period in months
        """
        self.base_config = base_config
        self.is_months = is_months
        self.oos_months = oos_months
        self.roll_months = roll_months
        
        self.result = WalkForwardResult()
    
    def generate_windows(self) -> list[WalkForwardWindow]:
        """Generate all walk-forward windows.
        
        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        window_id = 0
        
        current_start = self.base_config.start_date
        
        while True:
            # Calculate window boundaries
            is_end = current_start + timedelta(days=self.is_months * 30)
            oos_start = is_end
            oos_end = oos_start + timedelta(days=self.oos_months * 30)
            
            # Stop if OOS end exceeds backtest end date
            if oos_end > self.base_config.end_date:
                break
            
            window = WalkForwardWindow(
                window_id=window_id,
                is_start=current_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
            
            windows.append(window)
            window_id += 1
            
            # Roll forward
            current_start = current_start + timedelta(days=self.roll_months * 30)
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def label_regime(self, df: pd.DataFrame, start: datetime, end: datetime) -> str:
        """Label market regime for a period based on BTC price action.
        
        Regimes:
        - BULL: Drawdown < 20% and rally > 50% from start
        - BEAR: Drawdown > 20% from peak
        - CHOP: Neither bull nor bear criteria met
        
        Args:
            df: Price data with 'close' column
            start: Period start
            end: Period end
            
        Returns:
            Regime label string
        """
        # Filter to period
        mask = (df.index >= start) & (df.index <= end)
        period_df = df[mask]
        
        if len(period_df) == 0:
            return "UNKNOWN"
        
        closes = period_df['close'].values
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(closes)
        drawdown = (peak - closes) / peak
        max_dd = np.max(drawdown)
        
        # Calculate total return
        total_return = (closes[-1] - closes[0]) / closes[0]
        
        # Classify regime
        if max_dd > 0.20:
            return "BEAR"
        elif total_return > 0.50 and max_dd < 0.20:
            return "BULL"
        else:
            return "CHOP"
    
    async def run_window(
        self,
        window: WalkForwardWindow,
        data_dir,
        phase: str = "is",
    ) -> BacktestMetrics:
        """Run backtest for a single window phase.
        
        Args:
            window: Walk-forward window definition
            data_dir: Data directory path
            phase: 'is' for in-sample, 'oos' for out-of-sample
            
        Returns:
            BacktestMetrics for the phase
        """
        if phase == "is":
            config = BacktestConfig(
                start_date=window.is_start,
                end_date=window.is_end,
                symbols=self.base_config.symbols,
                timeframes=self.base_config.timeframes,
                initial_capital=self.base_config.initial_capital,
                risk_per_trade=self.base_config.risk_per_trade,
                is_major_assets=self.base_config.is_major_assets,
                fill_probability_major=self.base_config.fill_probability_major,
                fill_probability_alt=self.base_config.fill_probability_alt,
                maker_fee=self.base_config.maker_fee,
                taker_fee=self.base_config.taker_fee,
                slippage_major=self.base_config.slippage_major,
                slippage_alt=self.base_config.slippage_alt,
                seed=self.base_config.seed,
            )
        else:
            config = BacktestConfig(
                start_date=window.oos_start,
                end_date=window.oos_end,
                symbols=self.base_config.symbols,
                timeframes=self.base_config.timeframes,
                initial_capital=self.base_config.initial_capital,
                risk_per_trade=self.base_config.risk_per_trade,
                is_major_assets=self.base_config.is_major_assets,
                fill_probability_major=self.base_config.fill_probability_major,
                fill_probability_alt=self.base_config.fill_probability_alt,
                maker_fee=self.base_config.maker_fee,
                taker_fee=self.base_config.taker_fee,
                slippage_major=self.base_config.slippage_major,
                slippage_alt=self.base_config.slippage_alt,
                seed=self.base_config.seed,
            )
        
        engine = BacktestEngine(config)
        engine.load_data(data_dir)
        engine.add_universe_assets()
        
        # Inject signal generator if available
        # engine.signal_generator = self.signal_generator
        
        metrics = await engine.run()
        return metrics
    
    async def run(self, data_dir) -> WalkForwardResult:
        """Run complete walk-forward analysis.
        
        Args:
            data_dir: Path to historical data directory
            
        Returns:
            WalkForwardResult with all metrics
        """
        logger.info("Starting walk-forward validation")
        
        windows = self.generate_windows()
        
        # Load price data for regime labeling
        try:
            primary_symbol = self.base_config.symbols[0]
            primary_tf = self.base_config.timeframes[0]
            regime_df = pd.read_csv(
                f"{data_dir}/{primary_symbol}/{primary_tf}.csv",
                parse_dates=['timestamp'],
                index_col='timestamp',
            )
        except Exception as e:
            logger.warning(f"Could not load data for regime labeling: {e}")
            regime_df = None
        
        all_oos_equity = []
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Run in-sample
            logger.info(f"  Running IS: {window.is_start} to {window.is_end}")
            try:
                is_metrics = await self.run_window(window, data_dir, "is")
                window.is_metrics = is_metrics
                self.result.total_is_trades += is_metrics.total_trades
            except Exception as e:
                logger.error(f"  IS failed: {e}")
                continue
            
            # Run out-of-sample
            logger.info(f"  Running OOS: {window.oos_start} to {window.oos_end}")
            try:
                oos_metrics = await self.run_window(window, data_dir, "oos")
                window.oos_metrics = oos_metrics
                self.result.total_oos_trades += oos_metrics.total_trades
                
                # Stitch OOS equity curve
                if oos_metrics.equity_curve:
                    # Adjust starting equity to match end of previous window
                    if all_oos_equity:
                        last_equity = all_oos_equity[-1][1]
                        adjustment = last_equity / oos_metrics.equity_curve[0][1]
                        adjusted_equity = [(t, e * adjustment) for t, e in oos_metrics.equity_curve]
                        all_oos_equity.extend(adjusted_equity[1:])  # Skip first point (duplicate)
                    else:
                        all_oos_equity.extend(oos_metrics.equity_curve)
                        
            except Exception as e:
                logger.error(f"  OOS failed: {e}")
                continue
            
            # Label regime
            if regime_df is not None:
                regime = self.label_regime(regime_df, window.oos_start, window.oos_end)
                self.result.regime_labels[f"window_{window.window_id}"] = regime
                logger.info(f"  Regime: {regime}")
            
            self.result.windows.append(window)
        
        # Calculate aggregate metrics
        self.result.stitched_oos_equity = all_oos_equity
        
        valid_windows = [w for w in self.result.windows if w.is_metrics and w.oos_metrics]
        
        if valid_windows:
            self.result.avg_is_sharpe = np.mean([w.is_metrics.sharpe_ratio for w in valid_windows])
            self.result.avg_oos_sharpe = np.mean([w.oos_metrics.sharpe_ratio for w in valid_windows])
            self.result.avg_decay_ratio = np.mean([w.decay_ratio for w in valid_windows])
            
            overfitting_windows = sum(1 for w in valid_windows if w.is_overfitted)
            self.result.overfitting_windows = overfitting_windows
            self.result.overfitting_flagged = overfitting_windows > len(valid_windows) / 2
        
        logger.info(f"Walk-forward complete. Avg OOS Sharpe: {self.result.avg_oos_sharpe:.2f}")
        logger.info(f"Decay Ratio: {self.result.avg_decay_ratio:.2f}")
        logger.info(f"Overfitting Flagged: {self.result.overfitting_flagged}")
        
        return self.result
