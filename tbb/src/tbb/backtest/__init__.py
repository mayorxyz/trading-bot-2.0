"""Backtest module for event-driven strategy validation."""

from tbb.backtest.engine import BacktestEngine, BacktestConfig, BacktestMetrics, BacktestTrade
from tbb.backtest.costs import TransactionCostModel, OrderType, OrderSide, TransactionCostResult
from tbb.backtest.simulated_exchange import SimulatedExchange, BarData, Order, OrderStatus, OrderType as SimOrderType
from tbb.backtest.universe import UniverseManager, AssetInfo, UniverseState
from tbb.backtest.walk_forward import WalkForwardValidator, WalkForwardResult, WalkForwardWindow

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestTrade",
    "TransactionCostModel",
    "OrderType",
    "OrderSide",
    "TransactionCostResult",
    "SimulatedExchange",
    "BarData",
    "Order",
    "OrderStatus",
    "UniverseManager",
    "AssetInfo",
    "WalkForwardValidator",
    "WalkForwardResult",
]
