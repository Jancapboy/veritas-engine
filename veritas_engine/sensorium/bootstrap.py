"""Sensorium bootstrap — initializes and wires all sensorium components."""

from __future__ import annotations

import asyncio
from typing import Any

from veritas_engine.core.config import get_config
from veritas_engine.core.models import PerceptionEvent
from veritas_engine.core.logger import get_logger
from veritas_engine.sensorium.event_bus import EventBus
from veritas_engine.sensorium.anomaly_detector import AnomalyDetector
from veritas_engine.sensorium.base import BaseSensor, DigitalSensor, HeartbeatSensor, FileWatchSensor
from veritas_engine.sensorium.normalizer import DataNormalizer

logger = get_logger("veritas.sensorium")


class Sensorium:
    """感知层统一入口——管理所有传感器和事件总线."""

    def __init__(self) -> None:
        self.config = get_config().sensorium
        self.event_bus = EventBus()
        self.anomaly_detector = AnomalyDetector(
            zscore_threshold=self.config.anomaly_zscore_threshold,
            contamination=self.config.anomaly_contamination,
            n_estimators=self.config.anomaly_n_estimators,
        )
        self.sensors: list[BaseSensor] = []
        self._anomaly_sub_id: str | None = None

    async def start(self) -> None:
        """启动感知层：事件总线 + 传感器 + 异常检测订阅."""
        await self.event_bus.start()

        # 订阅所有事件进行异常检测
        self._anomaly_sub_id = await self.event_bus.subscribe(
            ["*"], self._anomaly_handler
        )

        # 启动默认传感器
        if self.config.enable_file_watch and self.config.watch_paths:
            for path in self.config.watch_paths:
                sensor = FileWatchSensor(f"file_{path}", path)
                await sensor.start(self._on_sensor_event)
                self.sensors.append(sensor)
        else:
            # 默认启动数字传感器和心跳传感器
            digital = DigitalSensor("digital_default", config={"interval": 2.0})
            await digital.start(self._on_sensor_event)
            self.sensors.append(digital)

            heartbeat = HeartbeatSensor("heartbeat", config={"interval": 5.0})
            await heartbeat.start(self._on_sensor_event)
            self.sensors.append(heartbeat)

        logger.info("Sensorium started with %d sensors", len(self.sensors), extra={"layer": "sensorium"})

    async def stop(self) -> None:
        """停止感知层."""
        for sensor in self.sensors:
            await sensor.stop()
        if self._anomaly_sub_id:
            await self.event_bus.unsubscribe(self._anomaly_sub_id)
        await self.event_bus.stop()
        logger.info("Sensorium stopped", extra={"layer": "sensorium"})

    async def emit(self, event: PerceptionEvent) -> None:
        """手动发布事件到总线."""
        await self.event_bus.emit(event)

    async def subscribe(self, event_types: list[str], handler: Any) -> str:
        """订阅事件."""
        return await self.event_bus.subscribe(event_types, handler)

    async def unsubscribe(self, subscription_id: str) -> None:
        """取消订阅."""
        await self.event_bus.unsubscribe(subscription_id)

    async def _on_sensor_event(self, event: PerceptionEvent) -> None:
        """传感器数据回调——转发到事件总线."""
        await self.event_bus.emit(event)

    async def _anomaly_handler(self, event: PerceptionEvent) -> None:
        """异常检测处理器——对所有数值型事件进行异常检测."""
        result = self.anomaly_detector.detect_event(event)
        if result and result["is_anomaly"]:
            anomaly_event = DataNormalizer.normalize_anomaly(
                source=event.source,
                anomaly_type="statistical_anomaly",
                details={
                    "original_event_id": event.id,
                    "score": result["score"],
                    "method": result["method"],
                    "value": event.payload.get("value"),
                },
                confidence=min(result["score"] / 10, 1.0),
            )
            await self.event_bus.emit(anomaly_event)
            logger.warning(
                "Anomaly detected: score=%.2f method=%s",
                result["score"],
                result["method"],
                extra={"layer": "sensorium"},
            )
