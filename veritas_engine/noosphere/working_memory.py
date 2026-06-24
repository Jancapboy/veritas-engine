"""Working memory — short-term memory window for recent events."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from veritas_engine.core.models import PerceptionEvent
from veritas_engine.core.config import get_config


class WorkingMemory:
    """工作记忆窗口——有限容量的短期记忆."""

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size or get_config().noosphere.working_memory_size
        self._events: deque[PerceptionEvent] = deque(maxlen=self.max_size)
        self._timestamp = datetime.now(timezone.utc)

    def add(self, event: PerceptionEvent) -> None:
        """添加事件到工作记忆."""
        self._events.append(event)

    def get_recent(self, n: int = 5) -> list[PerceptionEvent]:
        """获取最近 n 条事件."""
        return list(self._events)[-n:]

    def get_all(self) -> list[PerceptionEvent]:
        """获取所有工作记忆内容."""
        return list(self._events)

    def clear(self) -> None:
        """清空工作记忆."""
        self._events.clear()

    def summary(self) -> dict[str, Any]:
        """工作记忆摘要."""
        event_types = {}
        for e in self._events:
            et = e.event_type.value
            event_types[et] = event_types.get(et, 0) + 1
        return {
            "total_events": len(self._events),
            "max_size": self.max_size,
            "event_types": event_types,
            "oldest": self._events[0].timestamp.isoformat() if self._events else None,
            "newest": self._events[-1].timestamp.isoformat() if self._events else None,
        }
