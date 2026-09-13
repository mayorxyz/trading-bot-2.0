"""Simulated Exchange for Backtesting.

Implements realistic order filling logic including the "Conservative Price-Through Rule"
to avoid the limit order fill fallacy common in backtesters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import random


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents an order in the simulated exchange."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float                  # Limit price (None for market)
    stop_price: Optional[float]   # Stop trigger price (for stop orders)
    quantity: float               # Original quantity
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None  # Average fill price
    is_reduce_only: bool = False
    time_in_force: str = "GTC"    # GTC, IOC, FOK
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, or expired)."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]
    
    @property
    def fill_pct(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


@dataclass
class BarData:
    """OHLCV bar data for fill checking."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class SimulatedExchange:
    """Simulates exchange order matching with realistic fill logic.
    
    Implements the "Conservative Price-Through Rule":
    - Long limit fills only if bar.low < limit_price (strictly less, or by 1 tick)
    - Short limit fills only if bar.high > limit_price (strictly greater)
    - If price only touches but doesn't trade through, use fill_probability
    - Market orders always fill at next bar's open (with slippage applied separately)
    """
    
    # Minimum price movement threshold (1 tick for most crypto perps)
    TICK_SIZE = 0.5  # Adjust based on asset
    
    def __init__(
        self,
        fill_probability_major: float = 0.6,
        fill_probability_alt: float = 0.4,
        seed: Optional[int] = None,
    ):
        """Initialize simulated exchange.
        
        Args:
            fill_probability_major: Probability of fill when price touches limit (BTC/ETH)
            fill_probability_alt: Probability of fill when price touches limit (alts)
            seed: Random seed for reproducibility
        """
        self.fill_probability_major = fill_probability_major
        self.fill_probability_alt = fill_probability_alt
        self.orders: dict[str, Order] = {}
        self.filled_orders: list[Order] = []
        self.is_major_asset_cache: dict[str, bool] = {}
        
        if seed is not None:
            random.seed(seed)
    
    def is_major_asset(self, symbol: str) -> bool:
        """Check if symbol is a major asset (BTC/ETH)."""
        if symbol not in self.is_major_asset_cache:
            symbol_upper = symbol.upper()
            self.is_major_asset_cache[symbol_upper] = (
                'BTC' in symbol_upper or 'ETH' in symbol_upper
            )
        return self.is_major_asset_cache[symbol.upper()]
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: float = 0.0,
        is_reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> Order:
        """Create a new order.
        
        Args:
            symbol: Trading pair
            side: LONG or SHORT
            order_type: LIMIT, MARKET, STOP_MARKET, STOP_LIMIT
            price: Limit price (required for LIMIT/STOP_LIMIT)
            stop_price: Stop trigger price (required for STOP_*)
            quantity: Order size
            is_reduce_only: Whether order can only reduce position
            time_in_force: GTC, IOC, FOK
            
        Returns:
            Created Order object
        """
        order_id = f"ord_{symbol}_{datetime.utcnow().timestamp()}_{len(self.orders)}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            price=price if price is not None else 0.0,
            stop_price=stop_price,
            quantity=quantity,
            is_reduce_only=is_reduce_only,
            time_in_force=time_in_force,
        )
        
        self.orders[order_id] = order
        
        # Market orders execute immediately
        if order_type == OrderType.MARKET:
            # Will be filled on next bar
            pass
        
        return order
    
    def check_limit_fill(
        self,
        order: Order,
        bar: BarData,
    ) -> tuple[bool, Optional[float]]:
        """Check if a limit order should fill given bar data.
        
        Implements Conservative Price-Through Rule:
        - LONG: fills if bar.low < limit_price (strictly less)
        - SHORT: fills if bar.high > limit_price (strictly greater)
        - If only touches (bar.low == limit_price), use fill_probability
        
        Args:
            order: The limit order to check
            bar: Current bar data
            
        Returns:
            Tuple of (should_fill, fill_price)
        """
        if order.order_type != OrderType.LIMIT:
            return False, None
        
        limit_price = order.price
        fill_probability = (
            self.fill_probability_major if self.is_major_asset(order.symbol)
            else self.fill_probability_alt
        )
        
        if order.side == OrderSide.LONG:
            # Long limit: price must go BELOW limit to fill
            # Strict check: bar.low must be less than limit_price
            price_through = bar.low < limit_price
            price_touch = abs(bar.low - limit_price) < self.TICK_SIZE
            
            if price_through:
                # Definitely filled - price traded through our level
                fill_price = min(limit_price, bar.close)  # Better of limit or close
                return True, fill_price
            elif price_touch:
                # Price touched exactly - probabilistic fill
                if random.random() < fill_probability:
                    return True, limit_price
                else:
                    return False, None
            else:
                # Price never reached our level
                return False, None
                
        else:  # SHORT
            # Short limit: price must go ABOVE limit to fill
            price_through = bar.high > limit_price
            price_touch = abs(bar.high - limit_price) < self.TICK_SIZE
            
            if price_through:
                fill_price = max(limit_price, bar.close)
                return True, fill_price
            elif price_touch:
                if random.random() < fill_probability:
                    return True, limit_price
                else:
                    return False, None
            else:
                return False, None
    
    def check_stop_trigger(
        self,
        order: Order,
        bar: BarData,
    ) -> bool:
        """Check if a stop order should trigger.
        
        Args:
            order: The stop order to check
            bar: Current bar data
            
        Returns:
            True if stop should trigger
        """
        if order.stop_price is None:
            return False
        
        stop_price = order.stop_price
        
        if order.side == OrderSide.LONG:
            # Long stop triggers when price falls to stop level
            return bar.low <= stop_price
        else:
            # Short stop triggers when price rises to stop level
            return bar.high >= stop_price
    
    def process_bar(
        self,
        symbol: str,
        bar: BarData,
    ) -> list[Order]:
        """Process all orders for a symbol given bar data.
        
        Args:
            symbol: Trading pair symbol
            bar: Current bar data
            
        Returns:
            List of orders that were filled or updated
        """
        updated_orders = []
        symbol_upper = symbol.upper()
        
        for order_id, order in list(self.orders.items()):
            if order.symbol != symbol_upper or order.is_complete:
                continue
            
            # Check stop orders first
            if order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                if self.check_stop_trigger(order, bar):
                    # Stop triggered - convert to market/limit
                    if order.order_type == OrderType.STOP_MARKET:
                        # Will fill at next bar open
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = 0.0
                        order.status = OrderStatus.FILLED
                        order.fill_price = bar.open  # Simplified - slippage applied elsewhere
                        order.filled_at = bar.timestamp
                        self.filled_orders.append(order)
                        updated_orders.append(order)
                    else:  # STOP_LIMIT
                        # Now becomes a limit order
                        order.order_type = OrderType.LIMIT
                        # Continue to limit check below
                    continue
            
            # Check limit orders
            if order.order_type == OrderType.LIMIT:
                should_fill, fill_price = self.check_limit_fill(order, bar)
                
                if should_fill:
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = 0.0
                    order.status = OrderStatus.FILLED
                    order.fill_price = fill_price
                    order.filled_at = bar.timestamp
                    self.filled_orders.append(order)
                    updated_orders.append(order)
            
            # Market orders fill at bar open (next bar after placement)
            elif order.order_type == OrderType.MARKET and order.status == OrderStatus.PENDING:
                order.filled_quantity = order.quantity
                order.remaining_quantity = 0.0
                order.status = OrderStatus.FILLED
                order.fill_price = bar.open
                order.filled_at = bar.timestamp
                self.filled_orders.append(order)
                updated_orders.append(order)
        
        return updated_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.is_complete:
            return False
        
        order.status = OrderStatus.CANCELLED
        return True
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open (non-complete) orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        orders = [o for o in self.orders.values() if not o.is_complete]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
