"""
Exchange Client - Abstracts Bybit/Binance API calls via CCXT or direct REST.
Supports testnet routing, rate limiting, retry logic, and connection pooling.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

import httpx
from tbb.core.config import settings
from tbb.core.logger import setup_logger

logger = setup_logger(__name__)


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    LIMIT = "Limit"
    MARKET = "Market"


class OrderStatus(str, Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"
    UNTRIGGERED = "Untriggered"
    TRIGGERED = "Triggered"


class ExchangeClientError(Exception):
    """Base exception for exchange client errors."""
    pass


class RateLimitExceeded(ExchangeClientError):
    """Rate limit exceeded."""
    pass


class OrderNotFound(ExchangeClientError):
    """Order not found on exchange."""
    pass


class ExchangeClient:
    """
    Async exchange client supporting Bybit (primary) and Binance.
    Routes to testnet if IS_TESTNET=true.
    Implements retry logic, rate limiting, and connection pooling.
    """
    
    def __init__(self, is_testnet: bool = None, execution_mode: str = None):
        self.is_testnet = is_testnet if is_testnet is not None else settings.IS_TESTNET
        self.execution_mode = execution_mode or settings.EXECUTION_MODE
        self.api_key = settings.BYBIT_API_KEY
        self.api_secret = settings.BYBIT_API_SECRET
        
        # Rate limiting
        self.rate_limit_calls = 10  # per second for Bybit v5
        self.rate_limit_window = 1.0
        self._call_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Retry config
        self.max_retries = 3
        self.retry_delay = 0.5
        self.retry_backoff = 2.0
        
        # HTTP client with connection pooling
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Base URL
        self.base_url = settings.bybit_base_url
        if self.is_testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        logger.info(
            f"ExchangeClient initialized: testnet={self.is_testnet}, "
            f"mode={self.execution_mode}, base_url={self.base_url}"
        )
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._http_client
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        async with self._rate_limit_lock:
            now = time.time()
            # Remove calls outside window
            self._call_times = [t for t in self._call_times if now - t < self.rate_limit_window]
            
            if len(self._call_times) >= self.rate_limit_calls:
                sleep_time = self.rate_limit_window - (now - self._call_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self._call_times.append(time.time())
    
    async def _sign_request(self, method: str, path: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Generate signature for Bybit API request."""
        import hmac
        import hashlib
        from urllib.parse import urlencode
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if method == "GET":
            query_string = urlencode(params) if params else ""
            sign_str = f"{timestamp}{self.api_key}{recv_window}{query_string}"
        else:
            sign_str = f"{timestamp}{self.api_key}{recv_window}"
            if params:
                import json
                sign_str += json.dumps(params, separators=(',', ':'))
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to Bybit API with retry logic."""
        client = await self._get_http_client()
        url = f"{self.base_url}{endpoint}"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                
                headers = await self._sign_request(method, endpoint, data or params or {})
                
                if method == "GET":
                    response = await client.get(url, params=params, headers=headers)
                elif method == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method == "DELETE":
                    response = await client.delete(url, params=params, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                result = response.json()
                
                # Check Bybit-specific error codes
                if result.get("retCode") != 0:
                    error_msg = result.get("retMsg", "Unknown error")
                    if result.get("retCode") == 10001:  # Rate limit
                        raise RateLimitExceeded(f"Rate limit exceeded: {error_msg}")
                    raise ExchangeClientError(f"Bybit error {result.get('retCode')}: {error_msg}")
                
                return result.get("result", {})
            
            except (httpx.HTTPError, ExchangeClientError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise last_error
        
        raise last_error
    
    # === ORDER MANAGEMENT ===
    
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        qty: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """Place a limit order."""
        if self.execution_mode == "paper":
            logger.info(
                f"[PAPER MODE] Would place LIMIT {side.value} order: "
                f"{symbol} qty={qty} @ {price}, reduce_only={reduce_only}"
            )
            # Return mock response
            return {
                "orderId": f"PAPER_{int(time.time()*1000)}",
                "orderLinkId": "",
                "symbol": symbol,
                "side": side.value,
                "orderType": "Limit",
                "price": str(price),
                "qty": str(qty),
                "reduceOnly": reduce_only,
                "timeInForce": time_in_force,
                "orderStatus": "New",
            }
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.value,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "positionIdx": 1 if side == OrderSide.BUY else 2,
        }
        
        result = await self._request("POST", "/v5/order/create", data=params)
        logger.info(f"Limit order placed: {symbol} {side.value} {qty}@{price} | OrderID: {result.get('orderId')}")
        return result
    
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Place a market order (for emergency stops)."""
        if self.execution_mode == "paper":
            logger.info(
                f"[PAPER MODE] Would place MARKET {side.value} order: "
                f"{symbol} qty={qty}, reduce_only={reduce_only}"
            )
            return {
                "orderId": f"PAPER_MKT_{int(time.time()*1000)}",
                "symbol": symbol,
                "side": side.value,
                "orderType": "Market",
                "qty": str(qty),
                "reduceOnly": reduce_only,
                "orderStatus": "Filled",
            }
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.value,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": reduce_only,
            "positionIdx": 1 if side == OrderSide.BUY else 2,
        }
        
        result = await self._request("POST", "/v5/order/create", data=params)
        logger.info(f"Market order placed: {symbol} {side.value} {qty} | OrderID: {result.get('orderId')}")
        return result
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order by ID."""
        if self.execution_mode == "paper":
            logger.info(f"[PAPER MODE] Would cancel order {order_id} on {symbol}")
            return {"orderId": order_id, "symbol": symbol, "status": "Cancelled"}
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        
        result = await self._request("POST", "/v5/order/cancel", data=params)
        logger.info(f"Order cancelled: {symbol} {order_id}")
        return result
    
    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get open orders, optionally filtered by symbol or order ID."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        
        result = await self._request("GET", "/v5/order/realtime", params=params)
        orders = result.get("list", [])
        
        # Filter for only truly open orders
        open_statuses = {"NEW", "PARTIALLY_FILLED", "UNTRIGGERED"}
        open_orders = [o for o in orders if o.get("orderStatus") in open_statuses]
        
        return open_orders
    
    async def get_order_by_id(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get specific order by ID."""
        try:
            orders = await self.get_open_orders(symbol=symbol, order_id=order_id)
            if orders:
                return orders[0]
            
            # If not in open orders, check recent cancelled/filled
            params = {"category": "linear", "symbol": symbol, "orderId": order_id}
            result = await self._request("GET", "/v5/order/realtime", params=params)
            all_orders = result.get("list", [])
            if all_orders:
                return all_orders[0]
            
            return None
        except OrderNotFound:
            return None
    
    # === POSITION MANAGEMENT ===
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        params = {"category": "linear", "symbol": symbol}
        result = await self._request("GET", "/v5/position/list", params=params)
        positions = result.get("list", [])
        
        for pos in positions:
            if pos.get("symbol") == symbol and float(pos.get("size", 0)) != 0:
                return pos
        
        return None
    
    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        params = {"category": "linear", "settleCoin": "USDT"}
        result = await self._request("GET", "/v5/position/list", params=params)
        positions = result.get("list", [])
        
        # Filter for non-zero positions
        return [p for p in positions if float(p.get("size", 0)) != 0]
    
    async def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> Dict[str, Any]:
        """Close entire position for a symbol (market order)."""
        position = await self.get_position(symbol)
        if not position:
            logger.info(f"No open position for {symbol}")
            return {"status": "no_position"}
        
        size = float(position.get("size", 0))
        current_side = position.get("side", "")
        
        # Determine closing side
        if side:
            close_side = side
        else:
            close_side = OrderSide.SELL if current_side == "Buy" else OrderSide.BUY
        
        logger.info(f"Closing position: {symbol} {close_side.value} {size} (market)")
        return await self.place_market_order(symbol, close_side, size, reduce_only=True)
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        if self.execution_mode == "paper":
            logger.info(f"[PAPER MODE] Would cancel all orders for {symbol}")
            return {"symbol": symbol, "status": "cancelled_all"}
        
        params = {"category": "linear", "symbol": symbol}
        result = await self._request("POST", "/v5/order/cancel-all", data=params)
        logger.info(f"All orders cancelled for {symbol}")
        return result
    
    # === ACCOUNT INFO ===
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance (USDT wallet)."""
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        result = await self._request("GET", "/v5/account/wallet-balance", params=params)
        accounts = result.get("list", [])
        
        if accounts:
            coins = accounts[0].get("coin", [])
            for coin in coins:
                if coin.get("coin") == "USDT":
                    return {
                        "available": float(coin.get("availableToWithdraw", 0)),
                        "equity": float(coin.get("walletBalance", 0)),
                        "unrealized_pnl": float(coin.get("unrealisedPnl", 0)),
                    }
        
        return {"available": 0.0, "equity": 0.0, "unrealized_pnl": 0.0}
    
    async def get_daily_pnl(self, days: int = 1) -> float:
        """Get daily PnL for the last N days."""
        # Simplified: would need to fetch trade history and calculate
        # For MVP, return 0.0
        return 0.0
    
    # === KILL SWITCH ===
    
    async def emergency_kill_switch(self):
        """Cancel all orders and flatten all positions immediately."""
        logger.critical("KILL SWITCH ACTIVATED - Cancelling all orders and flattening positions")
        
        try:
            # Get all open positions
            positions = await self.get_all_positions()
            
            # Close all positions
            for pos in positions:
                symbol = pos.get("symbol")
                try:
                    await self.close_position(symbol)
                    logger.info(f"Emergency close completed for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to close {symbol}: {e}")
            
            # Cancel all orders for each symbol
            symbols = list(set(p.get("symbol") for p in positions))
            for symbol in symbols:
                try:
                    await self.cancel_all_orders(symbol)
                except Exception as e:
                    logger.error(f"Failed to cancel orders for {symbol}: {e}")
            
            logger.info("Kill switch execution completed")
        
        except Exception as e:
            logger.error(f"Kill switch failed: {e}")
            raise


# Singleton instance
_exchange_client: Optional[ExchangeClient] = None


def get_exchange_client() -> ExchangeClient:
    """Get or create exchange client singleton."""
    global _exchange_client
    if _exchange_client is None:
        _exchange_client = ExchangeClient()
    return _exchange_client


async def close_exchange_client():
    """Close exchange client HTTP session."""
    global _exchange_client
    if _exchange_client:
        await _exchange_client.close()
        _exchange_client = None
