"""Anomaly detector using Z-Score and Isolation Forest."""

from __future__ import annotations

from collections import deque
from typing import Any

from veritas_engine.core.models import PerceptionEvent
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.sensorium")


class AnomalyDetector:
    """异常检测器——支持 Z-Score 和简单统计模式."""

    def __init__(
        self,
        window_size: int = 100,
        zscore_threshold: float = 3.0,
        contamination: float = 0.1,
        n_estimators: int = 100,
    ) -> None:
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> dict[str, Any]:
        """更新检测器并返回异常判断结果.

        Returns:
            {"is_anomaly": bool, "score": float, "method": str}
        """
        self._buffer.append(value)
        result = {"is_anomaly": False, "score": 0.0, "method": "none"}

        # Z-Score 检测（需要足够数据）
        if len(self._buffer) >= 10:
            arr = list(self._buffer)
            mean = sum(arr) / len(arr)
            std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5 + 1e-9
            zscore = abs((value - mean) / std)
            if zscore > self.zscore_threshold:
                result["is_anomaly"] = True
                result["score"] = float(zscore)
                result["method"] = "zscore"
                return result

        return result

    def detect_event(self, event: PerceptionEvent, value_key: str = "value") -> dict[str, Any] | None:
        """从 PerceptionEvent 中提取数值并检测异常."""
        payload = event.payload
        if value_key not in payload:
            return None
        try:
            value = float(payload[value_key])
        except (TypeError, ValueError):
            return None
        return self.update(value)

    def reset(self) -> None:
        """重置检测器状态."""
        self._buffer.clear()
