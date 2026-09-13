"""Event-Driven Backtest Engine.

Mirrors the live paper-tracking state machine but runs on historical data.
Reuses existing signal generation logic from the live bot.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import polars as pl

from tbb.data.streamer import ParquetStreamer
from tbb.signals.market_structure import calculate_market_structure
from tbb.signals.confluence import score_confluence
from tbb.signals.generator import SignalGenerator, SignalResult
from tbb.tracking.store import SignalTrackingStore
from tbb.backtest.simulated_exchange import SimulatedExchange
from tbb.backtest.costs import TransactionCostModel
from tbb.core.config import settings
from tbb.core.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.streamer = ParquetStreamer()
        self.tracking_store = SignalTrackingStore()
        self.signal_generator = SignalGenerator(config={})
        self.cost_model = TransactionCostModel()
        self.results = {}

    async def run(self) -> Dict:
        """Main backtest loop using SAME functions as live runner."""
        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        
        all_signals = []
        all_trades = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            # 1. Stream data from Parquet (same source as live could use)
            async for batch in self.streamer.stream_range(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                interval="15m"
            ):
                # Convert PyArrow batch to Polars DataFrame
                df = pl.from_arrow(batch)
                
                # 2. SAME MARKET STRUCTURE CALCULATION AS LIVE
                df_struct = calculate_market_structure(df, config={})
                
                # 3. Iterate through candles (event-driven simulation)
                # Skip first 200 bars for warmup
                for i in range(200, len(df_struct)):
                    # Get historical context up to this point (NO LOOK-AHEAD)
                    historical_df = df_struct.slice(0, i+1)
                    
                    # 4. SAME CONFLUENCE SCORING AS LIVE
                    confluence_result = score_confluence(historical_df, config={})
                    
                    if confluence_result is not None:
                        # 5. SAME SIGNAL GENERATION AS LIVE
                        timestamp = historical_df["timestamp_utc"][i]
                        signal_result = self.signal_generator.generate_signal(
                            df=historical_df,
                            confluence_data=confluence_result,
                            symbol=symbol,
                            timestamp=timestamp
                        )
                        
                        if signal_result.signal is not None:
                            signal = signal_result.signal
                            
                            # 6. Record Signal (same as live tracker)
                            await self.tracking_store.add_signal(signal)
                            all_signals.append(signal)
                            
                            # 7. Simulate Execution (via SimulatedExchange)
                            exchange = SimulatedExchange(cost_model=self.cost_model)
                            trade = await exchange.simulate_signal_execution(signal, historical_df)
                            
                            if trade:
                                all_trades.append(trade)
                                logger.info(
                                    f"[{symbol}] Trade executed: {signal.direction} "
                                    f"@ {trade.entry_price}, Exit: {trade.exit_price}, "
                                    f"PnL: {trade.pnl_r}R"
                                )
        
        # 8. Calculate Metrics (same as live /tracking/stats)
        metrics = self._calculate_metrics(all_trades)
        
        self.results = {
            "signals": all_signals,
            "trades": all_trades,
            "metrics": metrics
        }
        
        return self.results

    def _calculate_metrics(self, trades: List) -> Dict:
        """Calculate same metrics as live /tracking/stats endpoint."""
        if not trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "total_trades": 0
            }
        
        wins = [t for t in trades if t.pnl_r > 0]
        losses = [t for t in trades if t.pnl_r < 0]
        
        total_win = sum(t.pnl_r for t in wins)
        total_loss = abs(sum(t.pnl_r for t in losses))
        
        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        expectancy_r = total_win / len(trades) if trades else 0
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "total_trades": len(trades),
            "avg_win_r": total_win / len(wins) if wins else 0,
            "avg_loss_r": total_loss / len(losses) if losses else 0
        }
    
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
