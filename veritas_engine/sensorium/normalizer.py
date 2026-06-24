"""Data normalizer — converts raw sensor data to standardized PerceptionEvent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from veritas_engine.core.models import PerceptionEvent, EventType


class DataNormalizer:
    """数据标准化器——将各种原始数据格式统一为 PerceptionEvent."""

    @staticmethod
    def normalize_opcua(
        node_id: str,
        value: Any,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PerceptionEvent:
        """标准化 OPC-UA 数据."""
        return PerceptionEvent(
            source="opcua",
            event_type=EventType.DATA_CHANGE,
            payload={"node_id": node_id, "value": value},
            metadata=metadata or {},
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    @staticmethod
    def normalize_mssql(
        table: str,
        operation: str,  # INSERT/UPDATE/DELETE
        row_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> PerceptionEvent:
        """标准化 MSSQL 变更数据."""
        return PerceptionEvent(
            source="mssql",
            event_type=EventType.DATA_CHANGE,
            payload={"table": table, "operation": operation, "row": row_data},
            metadata=metadata or {},
        )

    @staticmethod
    def normalize_file(
        path: str,
        content: str,
        change_type: str = "modified",
    ) -> PerceptionEvent:
        """标准化文件变更数据."""
        return PerceptionEvent(
            source="file",
            event_type=EventType.DATA_CHANGE,
            payload={"path": path, "change_type": change_type, "content_preview": content[:500]},
            metadata={},
        )

    @staticmethod
    def normalize_human_command(
        command: str,
        user: str = "anonymous",
        context: dict[str, Any] | None = None,
    ) -> PerceptionEvent:
        """标准化人类指令."""
        return PerceptionEvent(
            source="human",
            event_type=EventType.HUMAN_COMMAND,
            payload={"command": command, "user": user},
            metadata=context or {},
            confidence=1.0,
        )

    @staticmethod
    def normalize_anomaly(
        source: str,
        anomaly_type: str,
        details: dict[str, Any],
        confidence: float = 0.95,
    ) -> PerceptionEvent:
        """标准化异常事件."""
        return PerceptionEvent(
            source=source,
            event_type=EventType.ANOMALY,
            payload={"anomaly_type": anomaly_type, **details},
            confidence=confidence,
            metadata={},
        )
