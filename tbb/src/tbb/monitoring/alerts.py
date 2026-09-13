"""
Telegram Alert Module - Sends alerts via Telegram Bot API.
Stub implementation for MVP.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from enum import Enum

import httpx

from tbb.core.config import settings
from tbb.core.logger import setup_logger

logger = setup_logger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TelegramAlerter:
    """
    Sends alerts via Telegram Bot API.
    
    Alert levels:
    - CRITICAL: Immediate (circuit breaker, liquidation risk, CUSUM alert)
    - WARNING: Within 1h (Sharpe gap, win rate drop, consecutive losses 6-7)
    - DAILY_SUMMARY: EOD summary
    
    Anti-spam: 30-minute cooldown per alert type after first trigger.
    """
    
    COOLDOWN_MINUTES = 30
    
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self._last_alert_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured, alerts disabled")
        
        logger.info("TelegramAlerter initialized")
    
    async def send_alert(
        self,
        level: str,
        message: str,
        category: str,
    ) -> bool:
        """
        Send an alert via Telegram.
        
        Args:
            level: Alert level (CRITICAL, WARNING, INFO)
            message: Alert message (supports Markdown)
            category: Alert category for cooldown tracking
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Check cooldown
        if not await self._check_cooldown(category):
            logger.debug(f"Alert suppressed due to cooldown: {category}")
            return False
        
        # Send via Telegram API
        try:
            if not self.bot_token or not self.chat_id:
                logger.info(f"Telegram alert (mock): [{level}] {message[:100]}...")
                return True
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            
            # Record alert time
            async with self._lock:
                self._last_alert_times[category] = datetime.now(timezone.utc)
            
            logger.info(f"Telegram alert sent: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    async def _check_cooldown(self, category: str) -> bool:
        """Check if category is in cooldown period."""
        async with self._lock:
            last_time = self._last_alert_times.get(category)
            if last_time is None:
                return True
            
            now = datetime.now(timezone.utc)
            elapsed = (now - last_time).total_seconds() / 60
            
            return elapsed >= self.COOLDOWN_MINUTES
