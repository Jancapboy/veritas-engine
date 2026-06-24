"""Event bus using in-memory async pub/sub for Veritas Engine.

NATS embedded is complex; we use a lightweight in-memory event bus first.
Can be swapped to NATS later.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine
from uuid import uuid4

from veritas_engine.core.models import PerceptionEvent
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.sensorium")

SubscriptionHandler = Callable[[PerceptionEvent], Coroutine[Any, Any, None]]


class EventBus:
    """轻量级异步事件总线——内存发布/订阅."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, dict[str, SubscriptionHandler]] = {}
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[PerceptionEvent] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """启动事件总线分发器."""
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatcher())
        logger.info("EventBus started", extra={"layer": "sensorium"})

    async def stop(self) -> None:
        """停止事件总线."""
        self._running = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBus stopped", extra={"layer": "sensorium"})

    async def emit(self, event: PerceptionEvent) -> None:
        """发布事件到总线."""
        await self._event_queue.put(event)

    async def subscribe(
        self, event_types: list[str], handler: SubscriptionHandler
    ) -> str:
        """订阅特定类型的事件，返回订阅ID."""
        sub_id = str(uuid4())
        async with self._lock:
            for et in event_types:
                if et not in self._subscriptions:
                    self._subscriptions[et] = {}
                self._subscriptions[et][sub_id] = handler
        logger.debug(f"Subscribed [{sub_id}] to {event_types}", extra={"layer": "sensorium"})
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """取消订阅."""
        async with self._lock:
            for et, handlers in list(self._subscriptions.items()):
                handlers.pop(subscription_id, None)
                if not handlers:
                    del self._subscriptions[et]

    async def _dispatcher(self) -> None:
        """事件分发器——从队列取出事件并分发给订阅者."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            event_type = event.event_type.value
            handlers: dict[str, SubscriptionHandler] = {}
            async with self._lock:
                # 匹配具体类型和通配符 "*"
                handlers.update(self._subscriptions.get(event_type, {}))
                handlers.update(self._subscriptions.get("*", {}))

            for handler in handlers.values():
                try:
                    asyncio.create_task(handler(event))
                except Exception as e:
                    logger.error(f"Event handler error: {e}", extra={"layer": "sensorium"})
