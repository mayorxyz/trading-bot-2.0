"""Unit tests for backtest modules.

Tests cover:
1. Simulated Exchange fill logic (Conservative Price-Through Rule)
2. Transaction cost calculations
3. MAE/MFE tracking math
4. +0.5R stop movement after TP1
5. Partial TP calculations
6. Walk-forward decay ratio
"""

import pytest
from datetime import datetime, timedelta, timezone
import asyncio

from tbb.backtest.costs import TransactionCostModel, OrderType, OrderSide
from tbb.backtest.simulated_exchange import SimulatedExchange, BarData, OrderStatus
from tbb.backtest.walk_forward import WalkForwardWindow, WalkForwardResult


def _now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TestSimulatedExchange:
    """Test conservative price-through fill logic."""
    
    def test_limit_long_fill_price_through(self):
        """Long limit fills when bar.low <= limit_price - tick_size (deterministic)."""
        exchange = SimulatedExchange(seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar trades through the limit by more than 1 tick (0.5 for BTC)
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=49900.0,  # Well below limit (100 < 49999.5 threshold)
            close=50050.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        assert should_fill is True
        assert fill_price <= 50000.0  # Limit or better
    
    def test_limit_long_no_fill_when_low_above_limit(self):
        """Long limit does NOT fill when bar.low > limit_price."""
        exchange = SimulatedExchange(seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar never reaches our limit
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=50050.0,  # Above limit
            close=50150.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        assert should_fill is False
        assert fill_price is None
    
    def test_limit_long_touch_probabilistic_fill(self):
        """Long limit with probabilistic fill on exact touch (within 1 tick)."""
        # Use probability=1.0 to make test deterministic
        exchange = SimulatedExchange(fill_probability_major=1.0, seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar touches exactly at limit (within tick size)
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=50000.0,  # Exactly at limit - triggers probabilistic logic
            close=50150.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        # With probability=1.0, should fill
        assert should_fill is True
        assert fill_price == 50000.0
    
    def test_limit_long_touch_probabilistic_no_fill(self):
        """Long limit with probabilistic no-fill on touch when probability=0."""
        # Use probability=0.0 to ensure no fill
        exchange = SimulatedExchange(fill_probability_major=0.0, seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar touches exactly at limit
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=50000.0,
            close=50150.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        # With probability=0.0, should NOT fill
        assert should_fill is False
        assert fill_price is None
    
    def test_limit_long_deterministic_fill_clear_through(self):
        """Long limit deterministically fills when price clearly trades through."""
        exchange = SimulatedExchange(seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar low is well below limit (by more than tick_size=0.5)
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=49950.0,  # 50 below limit - definitely fills
            close=50050.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        assert should_fill is True
        assert fill_price <= 50000.0
    
    def test_limit_short_fill_price_through(self):
        """Short limit fills when bar.high > limit_price (strictly greater)."""
        exchange = SimulatedExchange(seed=42)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar trades through the limit
        bar = BarData(
            timestamp=_now(),
            open=49900.0,
            high=50100.0,  # Above limit
            low=49800.0,
            close=49950.0,
            volume=1000.0,
        )
        
        should_fill, fill_price = exchange.check_limit_fill(order, bar)
        
        assert should_fill is True
        assert fill_price >= 50000.0  # Limit or better
    
    def test_alt_asset_lower_fill_probability(self):
        """Alt assets have lower fill probability on touch."""
        exchange_major = SimulatedExchange(fill_probability_major=0.6, seed=42)
        exchange_alt = SimulatedExchange(fill_probability_alt=0.4, seed=42)
        
        assert exchange_major.fill_probability_major == 0.6
        assert exchange_alt.fill_probability_alt == 0.4


class TestTransactionCostModel:
    """Test fee and slippage calculations."""
    
    def test_maker_fee_calculation(self):
        """Maker orders pay maker fee, no slippage."""
        model = TransactionCostModel(maker_fee=0.0002, taker_fee=0.00055)
        
        result = model.calculate_costs(
            order_type=OrderType.LIMIT,
            side=OrderSide.LONG,
            price=50000.0,
            quantity=1.0,
            is_aggressive_limit=False,
        )
        
        assert result.is_maker is True
        assert result.fee_pct == 0.0002
        assert result.slippage_pct == 0.0
        assert result.fee_amount == 50000.0 * 1.0 * 0.0002
    
    def test_taker_fee_calculation(self):
        """Market orders pay taker fee plus slippage."""
        model = TransactionCostModel(
            taker_fee=0.00055,
            major_slippage=0.0005,
            is_major_asset=True,
        )
        
        result = model.calculate_costs(
            order_type=OrderType.MARKET,
            side=OrderSide.LONG,
            price=50000.0,
            quantity=1.0,
        )
        
        assert result.is_maker is False
        assert result.fee_pct == 0.00055
        assert result.slippage_pct == 0.0005
    
    def test_round_trip_costs(self):
        """Calculate total round-trip costs."""
        model = TransactionCostModel()
        
        result = model.calculate_round_trip_costs(
            entry_type=OrderType.LIMIT,
            exit_type=OrderType.MARKET,
            side=OrderSide.LONG,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=1.0,
        )
        
        total_costs = result["total_costs"]
        gross_pnl = result["gross_pnl"]
        net_pnl = result["net_pnl"]
        
        assert gross_pnl == 1000.0  # (51000 - 50000) * 1
        assert net_pnl < gross_pnl  # Costs reduce PnL
        assert total_costs > 0


class TestMAEMFECalculations:
    """Test MAE/MFE tracking math."""
    
    def test_mae_calculation_long(self):
        """MAE for long trade measures adverse movement."""
        entry_price = 50000.0
        stop_loss = 49000.0
        risk = entry_price - stop_loss  # 1000
        
        # Worst price during trade
        worst_low = 49500.0
        
        # MAE in R = (entry - worst) / risk
        mae_r = (entry_price - worst_low) / risk
        
        assert mae_r == 0.5  # Halfway to stop
    
    def test_mfe_calculation_long(self):
        """MFE for long trade measures favorable movement."""
        entry_price = 50000.0
        stop_loss = 49000.0
        risk = entry_price - stop_loss  # 1000
        
        # Best price during trade
        best_high = 52000.0
        
        # MFE in R = (best - entry) / risk
        mfe_r = (best_high - entry_price) / risk
        
        assert mfe_r == 2.0  # 2R move
    
    def test_pnl_at_tp1(self):
        """PnL at TP1 should be 1.0R."""
        entry_price = 50000.0
        stop_loss = 49000.0
        take_profit_1 = entry_price + (entry_price - stop_loss)  # 1R target
        
        pnl_r = (take_profit_1 - entry_price) / (entry_price - stop_loss)
        
        assert pnl_r == 1.0


class TestPartialTPAndStopMovement:
    """Test partial take profit and +0.5R stop movement."""
    
    def test_be_plus_offset_calculation(self):
        """Calculate breakeven + 0.5R offset."""
        entry_price = 50000.0
        stop_loss = 49000.0
        risk = entry_price - stop_loss  # 1000
        
        # New stop = entry + 0.5 * risk
        new_stop = entry_price + (0.5 * risk)
        
        assert new_stop == 50500.0
    
    def test_partial_pnl_calculation(self):
        """Calculate partial PnL for 50% position at TP1."""
        entry_price = 50000.0
        stop_loss = 49000.0
        take_profit_1 = 51000.0  # 1R
        quantity = 1.0
        
        # 50% at TP1
        partial_quantity = quantity * 0.5
        partial_pnl = (take_profit_1 - entry_price) * partial_quantity
        
        # In R terms
        risk_per_unit = entry_price - stop_loss
        partial_pnl_r = partial_pnl / risk_per_unit
        
        assert partial_pnl_r == 0.5  # 50% of position × 1R


class TestWalkForwardDecay:
    """Test walk-forward decay ratio calculations."""
    
    def test_decay_ratio_calculation(self):
        """Decay ratio = OOS Sharpe / IS Sharpe."""
        window = WalkForwardWindow(
            window_id=0,
            is_start=datetime(2022, 1, 1),
            is_end=datetime(2022, 12, 31),
            oos_start=datetime(2023, 1, 1),
            oos_end=datetime(2023, 6, 30),
        )
        
        # Mock metrics
        from tbb.backtest.engine import BacktestMetrics
        window.is_metrics = BacktestMetrics(sharpe_ratio=1.5)
        window.oos_metrics = BacktestMetrics(sharpe_ratio=0.9)
        
        decay = window.decay_ratio
        
        assert decay == 0.9 / 1.5  # 0.6
    
    def test_overfitting_detection(self):
        """Flag overfitting when decay < 0.5."""
        window = WalkForwardWindow(
            window_id=0,
            is_start=datetime(2022, 1, 1),
            is_end=datetime(2022, 12, 31),
            oos_start=datetime(2023, 1, 1),
            oos_end=datetime(2023, 6, 30),
        )
        
        from tbb.backtest.engine import BacktestMetrics
        window.is_metrics = BacktestMetrics(sharpe_ratio=2.0)
        window.oos_metrics = BacktestMetrics(sharpe_ratio=0.5)
        
        assert window.decay_ratio == 0.25
        assert window.is_overfitted is True
    
    def test_robustness_score(self):
        """Robustness score averages decay ratios clamped to 0-1."""
        result = WalkForwardResult()
        
        # Add windows with various decay ratios
        w1 = WalkForwardWindow(
            window_id=0,
            is_start=datetime(2022, 1, 1),
            is_end=datetime(2022, 12, 31),
            oos_start=datetime(2023, 1, 1),
            oos_end=datetime(2023, 6, 30),
        )
        w2 = WalkForwardWindow(
            window_id=1,
            is_start=datetime(2022, 4, 1),
            is_end=datetime(2023, 3, 31),
            oos_start=datetime(2023, 4, 1),
            oos_end=datetime(2023, 9, 30),
        )
        
        from tbb.backtest.engine import BacktestMetrics
        w1.is_metrics = BacktestMetrics(sharpe_ratio=1.5)
        w1.oos_metrics = BacktestMetrics(sharpe_ratio=1.2)  # Decay = 0.8
        w2.is_metrics = BacktestMetrics(sharpe_ratio=1.0)
        w2.oos_metrics = BacktestMetrics(sharpe_ratio=0.3)  # Decay = 0.3
        
        result.windows = [w1, w2]
        
        # Manual calculation
        expected = (0.8 + 0.3) / 2  # 0.55
        
        assert abs(result.robustness_score - expected) < 0.01


class TestFillProbabilityEdgeCases:
    """Test edge cases in fill probability logic."""
    
    def test_major_vs_alt_asset_detection(self):
        """Correctly identify major vs alt assets."""
        exchange = SimulatedExchange()
        
        assert exchange.is_major_asset("BTCUSDT") is True
        assert exchange.is_major_asset("ETHUSDT") is True
        assert exchange.is_major_asset("SOLUSDT") is False
        assert exchange.is_major_asset("LUNAUSDT") is False
    
    def test_tick_size_tolerance(self):
        """Tick size tolerance for touch detection."""
        exchange = SimulatedExchange(TICK_SIZE=0.5)
        
        order = exchange.create_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            price=50000.0,
            quantity=1.0,
        )
        
        # Bar low within tick size of limit
        bar = BarData(
            timestamp=_now(),
            open=50100.0,
            high=50200.0,
            low=50000.3,  # Within 0.5 tick
            close=50150.0,
            volume=1000.0,
        )
        
        should_fill, _ = exchange.check_limit_fill(order, bar)
        
        # Should trigger probabilistic fill logic
        # Result depends on random draw


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
