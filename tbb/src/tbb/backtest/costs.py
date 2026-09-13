"""Transaction Cost Model for Backtesting.

Implements realistic fee and slippage modeling based on 2024-2026 crypto perp research.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TransactionCostResult:
    """Result of transaction cost calculation."""
    fee_amount: float           # Absolute fee in quote currency
    fee_pct: float              # Fee as percentage
    slippage_amount: float      # Absolute slippage in quote currency
    slippage_pct: float         # Slippage as percentage
    total_cost: float           # Total cost (fee + slippage)
    effective_price: float      # Price after slippage
    is_maker: bool              # Whether order qualified as maker


class TransactionCostModel:
    """Realistic transaction cost modeling for crypto perps.
    
    Based on 2024-2026 microstructure research:
    - Maker fees: 0.02% (limit orders that rest)
    - Taker fees: 0.055% (market orders or aggressive limits)
    - Slippage: 0.05% for BTC/ETH, 0.15% for alts on market orders
    - Partial fills possible on limit orders during high volatility
    """
    
    # Default fee rates
    DEFAULT_MAKER_FEE = 0.0002      # 0.02%
    DEFAULT_TAKER_FEE = 0.00055     # 0.055%
    
    # Default slippage by asset type
    MAJOR_SLIPPAGE = 0.0005         # 0.05% for BTC/ETH
    ALT_SLIPPAGE = 0.0015           # 0.15% for altcoins
    
    def __init__(
        self,
        maker_fee: Optional[float] = None,
        taker_fee: Optional[float] = None,
        major_slippage: Optional[float] = None,
        alt_slippage: Optional[float] = None,
        is_major_asset: bool = True,
    ):
        """Initialize cost model.
        
        Args:
            maker_fee: Maker fee rate (default 0.02%)
            taker_fee: Taker fee rate (default 0.055%)
            major_slippage: Slippage for BTC/ETH (default 0.05%)
            alt_slippage: Slippage for alts (default 0.15%)
            is_major_asset: Whether this is a major asset (BTC/ETH)
        """
        self.maker_fee = maker_fee if maker_fee is not None else self.DEFAULT_MAKER_FEE
        self.taker_fee = taker_fee if taker_fee is not None else self.DEFAULT_TAKER_FEE
        self.major_slippage = major_slippage if major_slippage is not None else self.MAJOR_SLIPPAGE
        self.alt_slippage = alt_slippage if alt_slippage is not None else self.ALT_SLIPPAGE
        self.is_major_asset = is_major_asset
    
    def get_slippage_rate(self) -> float:
        """Get appropriate slippage rate for asset type."""
        return self.major_slippage if self.is_major_asset else self.alt_slippage
    
    def calculate_costs(
        self,
        order_type: OrderType,
        side: OrderSide,
        price: float,
        quantity: float,
        fill_price: Optional[float] = None,
        is_aggressive_limit: bool = False,
    ) -> TransactionCostResult:
        """Calculate transaction costs for an order.
        
        Args:
            order_type: LIMIT or MARKET
            side: LONG or SHORT
            price: Original order price
            quantity: Order quantity
            fill_price: Actual fill price (if different from order price)
            is_aggressive_limit: Whether limit order was aggressive (taker)
            
        Returns:
            TransactionCostResult with all cost components
        """
        effective_price = fill_price if fill_price is not None else price
        
        # Determine if maker or taker
        # Market orders are always taker
        # Limit orders are maker UNLESS they're aggressive (immediate fill)
        is_maker = (order_type == OrderType.LIMIT and not is_aggressive_limit)
        
        # Select fee rate
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        
        # Calculate slippage (only for taker/market orders)
        # For maker orders, slippage is typically zero or negative (price improvement)
        if is_maker:
            slippage_rate = 0.0
        else:
            slippage_rate = self.get_slippage_rate()
        
        # Apply slippage direction based on side
        if side == OrderSide.LONG:
            # Longs pay more due to slippage
            slippage_adjustment = 1.0 + slippage_rate
        else:
            # Shorts receive less due to slippage
            slippage_adjustment = 1.0 - slippage_rate
        
        # Adjust effective price for slippage (only for taker)
        if not is_maker and fill_price is None:
            effective_price = price * slippage_adjustment
        
        # Calculate costs
        notional = effective_price * quantity
        fee_amount = notional * fee_rate
        slippage_amount = abs(effective_price - price) * quantity
        total_cost = fee_amount + slippage_amount
        
        return TransactionCostResult(
            fee_amount=fee_amount,
            fee_pct=fee_rate,
            slippage_amount=slippage_amount,
            slippage_pct=slippage_rate if not is_maker else 0.0,
            total_cost=total_cost,
            effective_price=effective_price,
            is_maker=is_maker,
        )
    
    def calculate_round_trip_costs(
        self,
        entry_type: OrderType,
        exit_type: OrderType,
        side: OrderSide,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> dict:
        """Calculate total round-trip transaction costs.
        
        Args:
            entry_type: LIMIT or MARKET for entry
            exit_type: LIMIT or MARKET for exit
            side: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            
        Returns:
            Dict with entry_costs, exit_costs, total_costs, net_pnl_impact
        """
        entry_costs = self.calculate_costs(entry_type, side, entry_price, quantity)
        exit_costs = self.calculate_costs(exit_type, side, exit_price, quantity)
        
        total_costs = entry_costs.total_cost + exit_costs.total_cost
        
        # Calculate PnL impact
        if side == OrderSide.LONG:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        net_pnl = gross_pnl - total_costs
        
        return {
            "entry_costs": entry_costs,
            "exit_costs": exit_costs,
            "total_costs": total_costs,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "costs_as_pct_of_gross_pnl": (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else float('inf'),
        }
