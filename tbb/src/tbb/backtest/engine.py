"""Event-Driven Backtest Engine.

Mirrors the live paper-tracking state machine but runs on historical data.
Reuses existing signal generation logic from the live bot.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
import pandas as pd

from tbb.core.config import settings
from tbb.core.logger import get_logger
from tbb.backtest.costs import TransactionCostModel, OrderType, OrderSide
from tbb.backtest.simulated_exchange import SimulatedExchange, BarData, Order as SimOrder
from tbb.backtest.universe import UniverseManager
from tbb.tracking.store import TrackedSignal as SignalRecord, TrackingStatus
from tbb.tracking.resolver import SignalResolver

logger = get_logger(__name__)


class BacktestStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class BacktestTrade:
    """Represents a completed trade in backtest."""
    signal_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    pnl_r: float
    pnl_pct: float
    mae_r: float
    mfe_r: float
    fees_paid: float
    slippage_paid: float
    partial_pnl_r: float = 0.0
    runner_pnl_r: float = 0.0


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_pnl_r: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_trade_duration_hours: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    mae_median_winners: float = 0.0
    mfe_capture_rate: float = 0.0
    early_exit_rate: float = 0.0
    invalidation_rate: float = 0.0
    fill_rate: float = 0.0
    avg_fees_paid: float = 0.0
    avg_slippage_paid: float = 0.0
    
    # Walk-forward metrics
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_vs_oos_decay: float = 1.0
    regime_labels: dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    timeframes: list[str] = field(default_factory=lambda: ["15m", "1H", "4H"])
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01
    is_major_assets: bool = True
    fill_probability_major: float = 0.6
    fill_probability_alt: float = 0.4
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055
    slippage_major: float = 0.0005
    slippage_alt: float = 0.0015
    seed: Optional[int] = None
    enable_walk_forward: bool = False
    is_months: int = 12
    oos_months: int = 6
    roll_months: int = 3


class BacktestEngine:
    """Event-driven backtest engine mirroring live state machine.
    
    Architecture:
    1. Loads historical OHLCV data
    2. Iterates through bars chronologically
    3. At each bar, runs signal generation (reusing live logic)
    4. Simulated exchange processes orders with realistic fill logic
    5. Cost model applies fees and slippage
    6. Resolver tracks MAE/MFE and manages exits
    7. Records all trades and calculates metrics
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.status = BacktestStatus.NOT_STARTED
        
        # Initialize components
        self.exchange = SimulatedExchange(
            fill_probability_major=config.fill_probability_major,
            fill_probability_alt=config.fill_probability_alt,
            seed=config.seed,
        )
        
        self.cost_model = TransactionCostModel(
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
            major_slippage=config.slippage_major,
            alt_slippage=config.slippage_alt,
            is_major_asset=config.is_major_assets,
        )
        
        self.universe = UniverseManager()
        
        # State tracking
        self.current_time: Optional[datetime] = None
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[BacktestTrade] = []
        self.open_signals: dict[str, SignalRecord] = {}
        self.pending_orders: dict[str, SimOrder] = {}
        
        # Data storage
        self.price_data: dict[str, dict[str, pd.DataFrame]] = {}  # symbol -> timeframe -> df
        
        # Metrics calculation
        self.equity_high_watermark = config.initial_capital
        self.max_drawdown = 0.0
        
        # Callbacks for signal generation (injected from live modules)
        self.signal_generator: Optional[Callable] = None
        self.state_machine: Optional[SignalResolver] = None
    
    def load_data(self, data_dir: Path):
        """Load historical OHLCV data from files.
        
        Expected structure:
        data_dir/
          BTCUSDT/
            15m.csv
            1H.csv
            4H.csv
          ETHUSDT/
            ...
        
        Args:
            data_dir: Directory containing historical data
        """
        logger.info(f"Loading historical data from {data_dir}")
        
        for symbol in self.config.symbols:
            self.price_data[symbol] = {}
            
            for timeframe in self.config.timeframes:
                filepath = data_dir / symbol / f"{timeframe}.csv"
                
                if not filepath.exists():
                    logger.warning(f"Data file not found: {filepath}")
                    continue
                
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                df = df.set_index('timestamp')
                
                # Filter to backtest date range
                mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                df = df[mask]
                
                if len(df) == 0:
                    logger.warning(f"No data in range for {symbol} {timeframe}")
                    continue
                
                self.price_data[symbol][timeframe] = df
                logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
    
    def add_universe_assets(self):
        """Add assets to universe manager (for survivorship bias handling)."""
        # Default: assume all assets are active throughout backtest
        # Can be overridden by loading from CSV
        for symbol in self.config.symbols:
            self.universe.add_asset(
                symbol=symbol,
                name=symbol,
                listed_date=self.config.start_date - timedelta(days=365),
                delisted_date=None,
            )
    
    async def run(self) -> BacktestMetrics:
        """Run the backtest.
        
        Returns:
            BacktestMetrics with all performance statistics
        """
        logger.info("Starting backtest")
        self.status = BacktestStatus.RUNNING
        
        current_equity = self.config.initial_capital
        self.equity_curve.append((self.config.start_date, current_equity))
        
        # Get primary timeframe for iteration (usually the signal timeframe)
        primary_tf = self.config.timeframes[0]  # e.g., "15m"
        
        # Iterate through all bars chronologically
        # For multi-symbol, we need to merge timestamps
        all_timestamps = set()
        for symbol in self.config.symbols:
            if primary_tf in self.price_data.get(symbol, {}):
                df = self.price_data[symbol][primary_tf]
                all_timestamps.update(df.index.tolist())
        
        sorted_timestamps = sorted(all_timestamps)
        
        logger.info(f"Processing {len(sorted_timestamps)} bars")
        
        for i, timestamp in enumerate(sorted_timestamps):
            self.current_time = timestamp
            
            # Check for delistings
            delistings = self.universe.check_delistings(timestamp)
            for asset in delistings:
                await self._handle_delisting(asset)
            
            # Process each symbol
            for symbol in self.config.symbols:
                if not self.universe.is_tradable(symbol, timestamp):
                    continue
                
                if symbol not in self.price_data or primary_tf not in self.price_data[symbol]:
                    continue
                
                df = self.price_data[symbol][primary_tf]
                if timestamp not in df.index:
                    continue
                
                bar = df.loc[timestamp]
                
                # Create BarData for simulated exchange
                bar_data = BarData(
                    timestamp=timestamp,
                    open=float(bar['open']),
                    high=float(bar['high']),
                    low=float(bar['low']),
                    close=float(bar['close']),
                    volume=float(bar['volume']),
                )
                
                # Step 1: Generate signals (if enabled)
                if self.signal_generator:
                    await self._generate_signals(symbol, bar_data)
                
                # Step 2: Process orders through simulated exchange
                filled_orders = self.exchange.process_bar(symbol, bar_data)
                
                # Step 3: Handle filled orders
                for order in filled_orders:
                    await self._handle_order_fill(order, bar_data)
                
                # Step 4: Update open positions (MAE/MFE tracking)
                await self._update_open_positions(symbol, bar_data)
                
                # Step 5: Check for exit conditions
                await self._check_exits(symbol, bar_data)
            
            # Update equity curve
            current_equity = self._calculate_current_equity()
            self.equity_curve.append((timestamp, current_equity))
            
            # Update drawdown
            if current_equity > self.equity_high_watermark:
                self.equity_high_watermark = current_equity
            drawdown = (self.equity_high_watermark - current_equity) / self.equity_high_watermark
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(sorted_timestamps)} bars, Equity: ${current_equity:.2f}")
        
        # Finalize
        self.status = BacktestStatus.COMPLETED
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        # Calculate and return metrics
        return self._calculate_metrics()
    
    async def _generate_signals(self, symbol: str, bar: BarData):
        """Generate signals using live signal generator logic.
        
        This should call the same signal generation code as the live bot.
        For now, this is a placeholder for integration.
        """
        if not self.signal_generator:
            return
        
        # Call signal generator with current bar data
        # The signal generator should return MVSResult or similar
        try:
            signals = await self.signal_generator(symbol, bar, self.current_time)
            
            for signal in signals:
                if signal and signal.signal_valid:
                    await self._process_new_signal(signal)
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
    
    async def _process_new_signal(self, signal):
        """Process a newly generated signal."""
        # Create SignalRecord
        record = SignalRecord(
            signal_id=f"bt_{signal.symbol}_{self.current_time.timestamp()}",
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_price,
            take_profit_1=signal.target_price,  # Will be calculated properly in real impl
            take_profit_2=signal.target_price * 1.5,  # Example TP2
            confluence_score=signal.confidence,
            timeframe="15m",  # From signal
            created_at=self.current_time,
            tracking_status=TrackingStatus.NEW,
        )
        
        self.open_signals[record.signal_id] = record
        
        # If auto-execute, place entry order
        if settings.AUTO_EXECUTE:
            order = self._create_entry_order(record)
            if order:
                self.pending_orders[record.signal_id] = order
    
    def _create_entry_order(self, signal: SignalRecord) -> Optional[SimOrder]:
        """Create entry order for signal."""
        side = OrderSide.LONG if signal.direction == "LONG" else OrderSide.SHORT
        
        order = self.exchange.create_order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=signal.entry_price,
            quantity=signal.position_size_notional / signal.entry_price if signal.position_size_notional else 1.0,
            is_reduce_only=False,
        )
        
        return order
    
    async def _handle_order_fill(self, order: SimOrder, bar: BarData):
        """Handle when an order is filled."""
        # Find corresponding signal
        signal_record = None
        for sig_id, sig in self.open_signals.items():
            if sig.signal_id in order.order_id or sig_id in order.order_id:
                signal_record = sig
                break
        
        if not signal_record:
            return
        
        # Calculate transaction costs
        side = OrderSide.LONG if signal_record.direction == "LONG" else OrderSide.SHORT
        costs = self.cost_model.calculate_costs(
            order_type=OrderType.LIMIT,  # Entry is limit
            side=side,
            price=order.price,
            quantity=order.filled_quantity,
            fill_price=order.fill_price,
        )
        
        # Update signal record
        signal_record.entry_fill_price = float(order.fill_price)
        signal_record.entry_filled_at = bar.timestamp
        signal_record.tracking_status = TrackingStatus.ACTIVE
        
        # Store costs
        signal_record.metadata = signal_record.metadata or {}
        signal_record.metadata['entry_fees'] = costs.fee_amount
        signal_record.metadata['entry_slippage'] = costs.slippage_amount
    
    async def _update_open_positions(self, symbol: str, bar: BarData):
        """Update MAE/MFE for open positions."""
        for signal_id, signal in list(self.open_signals.items()):
            if signal.tracking_status != TrackingStatus.ACTIVE:
                continue
            if signal.symbol != symbol:
                continue
            
            # Calculate current PnL in R
            if signal.direction == "LONG":
                current_pnl_r = (bar.close - signal.entry_fill_price) / (signal.entry_fill_price - signal.stop_loss)
                mfe_price = max(bar.high, bar.close)
                mae_price = min(bar.low, bar.close)
            else:
                current_pnl_r = (signal.entry_fill_price - bar.close) / (signal.entry_fill_price - signal.stop_loss)
                mfe_price = max(bar.high, bar.close)
                mae_price = min(bar.low, bar.close)
            
            # Update MFE
            if signal.mfe_r is None or current_pnl_r > signal.mfe_r:
                signal.mfe_r = current_pnl_r
            
            # Update MAE
            mae_r = abs(min(0, current_pnl_r))  # Only adverse movement
            if signal.mae_r is None or mae_r > signal.mae_r:
                signal.mae_r = mae_r
    
    async def _check_exits(self, symbol: str, bar: BarData):
        """Check for exit conditions on open positions."""
        for signal_id, signal in list(self.open_signals.items()):
            if signal.tracking_status != TrackingStatus.ACTIVE:
                continue
            if signal.symbol != symbol:
                continue
            
            # Check TP1 hit
            tp1_hit = self._check_tp1_hit(signal, bar)
            if tp1_hit and signal.tracking_status != TrackingStatus.HIT_TP1:
                signal.tracking_status = TrackingStatus.HIT_TP1
                signal.partial_pnl_r = 1.0  # 50% at 1R
                # Move stop to +0.5R
                new_stop = self._calculate_be_plus_offset(signal)
                signal.stop_loss = new_stop
                continue
            
            # Check TP2 hit
            if signal.take_profit_2:
                if signal.direction == "LONG" and bar.high >= signal.take_profit_2:
                    await self._close_position(signal, bar, "TAKE_PROFIT_2")
                    continue
                elif signal.direction == "SHORT" and bar.low <= signal.take_profit_2:
                    await self._close_position(signal, bar, "TAKE_PROFIT_2")
                    continue
            
            # Check SL hit
            if signal.direction == "LONG" and bar.low <= signal.stop_loss:
                await self._close_position(signal, bar, "STOP_LOSS")
                continue
            elif signal.direction == "SHORT" and bar.high >= signal.stop_loss:
                await self._close_position(signal, bar, "STOP_LOSS")
                continue
            
            # Check MAE early exit
            if signal.mae_r and signal.mae_r > 1.0:
                # Calculate current PnL
                if signal.direction == "LONG":
                    current_pnl = (bar.close - signal.entry_fill_price) / (signal.entry_fill_price - signal.stop_loss)
                else:
                    current_pnl = (signal.entry_fill_price - bar.close) / (signal.entry_fill_price - signal.stop_loss)
                
                if current_pnl < 0:
                    await self._close_position(signal, bar, "EARLY_EXIT_MAE")
                    continue
            
            # Check expiry
            if signal.expiry_time and bar.timestamp > signal.expiry_time:
                if signal.tracking_status == TrackingStatus.NEW:
                    signal.tracking_status = TrackingStatus.EXPIRED
                elif signal.tracking_status == TrackingStatus.ACTIVE:
                    await self._close_position(signal, bar, "TIME_EXIT")
    
    def _check_tp1_hit(self, signal: SignalRecord, bar: BarData) -> bool:
        """Check if TP1 was hit."""
        if signal.direction == "LONG":
            return bar.high >= signal.take_profit_1
        else:
            return bar.low <= signal.take_profit_1
    
    def _calculate_be_plus_offset(self, signal: SignalRecord) -> float:
        """Calculate breakeven + 0.5R offset."""
        risk_r = abs(signal.entry_fill_price - signal.stop_loss)
        offset = 0.5 * risk_r
        
        if signal.direction == "LONG":
            return signal.entry_fill_price + offset
        else:
            return signal.entry_fill_price - offset
    
    async def _close_position(
        self,
        signal: SignalRecord,
        bar: BarData,
        reason: str,
    ):
        """Close a position and record the trade."""
        # Determine exit price
        if reason == "STOP_LOSS":
            exit_price = signal.stop_loss
        elif reason.startswith("TAKE_PROFIT"):
            exit_price = signal.take_profit_1 if reason == "TAKE_PROFIT_1" else signal.take_profit_2
        else:
            exit_price = bar.close  # Market exit
        
        # Calculate PnL
        if signal.direction == "LONG":
            pnl_r = (exit_price - signal.entry_fill_price) / (signal.entry_fill_price - signal.stop_loss)
        else:
            pnl_r = (signal.entry_fill_price - exit_price) / (signal.entry_fill_price - signal.stop_loss)
        
        # Calculate costs for exit (market order = taker)
        side = OrderSide.LONG if signal.direction == "LONG" else OrderSide.SHORT
        quantity = signal.position_size_notional / signal.entry_fill_price if signal.position_size_notional else 1.0
        
        exit_costs = self.cost_model.calculate_costs(
            order_type=OrderType.MARKET,
            side=side,
            price=exit_price,
            quantity=quantity,
        )
        
        # Adjust PnL for costs
        total_costs = signal.metadata.get('entry_fees', 0) + exit_costs.fee_amount
        pnl_r_adjusted = pnl_r - (total_costs / (signal.entry_fill_price - signal.stop_loss))
        
        # Create trade record
        trade = BacktestTrade(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.direction,
            entry_price=signal.entry_fill_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_time=signal.entry_filled_at,
            exit_time=bar.timestamp,
            exit_reason=reason,
            pnl_r=pnl_r_adjusted,
            pnl_pct=pnl_r_adjusted * ((signal.entry_fill_price - signal.stop_loss) / signal.entry_fill_price),
            mae_r=signal.mae_r or 0.0,
            mfe_r=signal.mfe_r or 0.0,
            fees_paid=total_costs,
            slippage_paid=signal.metadata.get('entry_slippage', 0) + exit_costs.slippage_amount,
            partial_pnl_r=signal.partial_pnl_r or 0.0,
            runner_pnl_r=(pnl_r_adjusted - signal.partial_pnl_r) if signal.partial_pnl_r else pnl_r_adjusted,
        )
        
        self.trades.append(trade)
        
        # Update signal record
        signal.tracking_status = TrackingStatus.RESOLVED
        signal.exit_price = exit_price
        signal.exit_reason = reason
        signal.pnl_r = pnl_r_adjusted
        signal.resolved_at = bar.timestamp
        
        logger.info(f"Closed {signal.symbol} {signal.direction}: {reason}, PnL: {pnl_r_adjusted:.2f}R")
    
    async def _handle_delisting(self, asset):
        """Handle asset delisting - force close positions."""
        terminal_value = self.universe.get_terminal_value(asset.symbol)
        
        for signal_id, signal in list(self.open_signals.items()):
            if signal.symbol == asset.symbol and signal.tracking_status == TrackingStatus.ACTIVE:
                # Force close at terminal value
                loss_pct = 1.0 - terminal_value  # e.g., 1.0 - 0.0 = -100%
                signal.tracking_status = TrackingStatus.INVALIDATED
                signal.exit_reason = f"DELISTED: {asset.delisting_reason}"
                
                logger.warning(f"Force closed {signal.symbol} due to delisting: {asset.delisting_reason}")
    
    def _calculate_current_equity(self) -> float:
        """Calculate current equity including open positions."""
        equity = self.config.initial_capital
        
        # Add realized PnL from closed trades
        for trade in self.trades:
            equity += trade.pnl_r * (trade.entry_price - trade.exit_price)  # Simplified
        
        return equity
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate final backtest metrics."""
        metrics = BacktestMetrics()
        
        if not self.trades:
            return metrics
        
        # Basic stats
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.pnl_r > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.pnl_r <= 0)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        # PnL stats
        total_pnl_r = sum(t.pnl_r for t in self.trades)
        metrics.total_pnl_r = total_pnl_r
        metrics.expectancy_r = total_pnl_r / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        gross_profit = sum(t.pnl_r for t in self.trades if t.pnl_r > 0)
        gross_loss = abs(sum(t.pnl_r for t in self.trades if t.pnl_r < 0))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # MAE/MFE stats
        metrics.avg_mae_r = sum(t.mae_r for t in self.trades) / len(self.trades)
        metrics.avg_mfe_r = sum(t.mfe_r for t in self.trades) / len(self.trades)
        
        winners = [t for t in self.trades if t.pnl_r > 0]
        if winners:
            metrics.mae_median_winners = sorted(t.mae_r for t in winners)[len(winners)//2]
            metrics.mfe_capture_rate = sum(t.pnl_r for t in winners) / sum(t.mfe_r for t in winners) if sum(t.mfe_r for t in winners) > 0 else 0.0
        
        # Early exit and invalidation rates
        early_exits = sum(1 for t in self.trades if 'EARLY_EXIT' in t.exit_reason)
        metrics.early_exit_rate = early_exits / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        invalidations = sum(1 for t in self.trades if 'INVALIDATED' in t.exit_reason or 'EXPIRED' in t.exit_reason)
        metrics.invalidation_rate = invalidations / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        # Costs
        metrics.avg_fees_paid = sum(t.fees_paid for t in self.trades) / len(self.trades)
        metrics.avg_slippage_paid = sum(t.slippage_paid for t in self.trades) / len(self.trades)
        
        # Duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
        metrics.avg_trade_duration_hours = sum(durations) / len(durations) if durations else 0.0
        
        # Drawdown (already tracked during run)
        metrics.max_drawdown = self.max_drawdown
        
        # Sharpe ratio (simplified - using R returns)
        if len(self.trades) > 1:
            import numpy as np
            returns = [t.pnl_r for t in self.trades]
            metrics.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        
        # Fill rate
        filled_orders = len(self.trades)
        total_orders = filled_orders + len([s for s in self.open_signals.values() if s.tracking_status == TrackingStatus.EXPIRED])
        metrics.fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
        
        return metrics
