"""Sensor base class and digital sensor implementations for Veritas Engine."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Coroutine

from veritas_engine.core.models import PerceptionEvent
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.sensorium")


class BaseSensor(ABC):
    """传感器抽象基类——所有传感器的统一接口."""

    def __init__(self, name: str, source: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.source = source
        self.config = config or {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._callback: Callable[[PerceptionEvent], Coroutine[Any, Any, None]] | None = None

    @abstractmethod
    async def _read(self) -> PerceptionEvent | None:
        """读取一次传感器数据，返回感知事件或None."""

    @abstractmethod
    async def _open(self) -> None:
        """打开传感器连接/资源."""

    @abstractmethod
    async def _close(self) -> None:
        """关闭传感器连接/资源."""

    async def start(self, callback: Callable[[PerceptionEvent], Coroutine[Any, Any, None]]) -> None:
        """启动传感器，设置数据回调."""
        self._callback = callback
        await self._open()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Sensor [{self.name}] started", extra={"layer": "sensorium"})

    async def stop(self) -> None:
        """停止传感器."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._close()
        logger.info(f"Sensor [{self.name}] stopped", extra={"layer": "sensorium"})

    async def _loop(self) -> None:
        """主循环：定期读取数据并回调."""
        interval = self.config.get("interval", 1.0)
        while self._running:
            try:
                event = await self._read()
                if event and self._callback:
                    await self._callback(event)
            except Exception as e:
                logger.error(f"Sensor [{self.name}] read error: {e}", extra={"layer": "sensorium"})
            await asyncio.sleep(interval)


class FileWatchSensor(BaseSensor):
    """文件监控传感器——监控文件变化."""

    def __init__(self, name: str, watch_path: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, "file", config)
        self.watch_path = Path(watch_path)
        self._last_mtime: float | None = None

    async def _open(self) -> None:
        if not self.watch_path.exists():
            self.watch_path.parent.mkdir(parents=True, exist_ok=True)
            self.watch_path.write_text("")

    async def _close(self) -> None:
        pass

    async def _read(self) -> PerceptionEvent | None:
        if not self.watch_path.exists():
            return None
        mtime = self.watch_path.stat().st_mtime
        if self._last_mtime is None:
            self._last_mtime = mtime
            return None
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            content = self.watch_path.read_text(encoding="utf-8", errors="ignore")
            return PerceptionEvent(
                source="file",
                event_type="data_change",
                payload={
                    "path": str(self.watch_path),
                    "mtime": mtime,
                    "size": self.watch_path.stat().st_size,
                    "content_preview": content[:500],
                },
                metadata={"sensor_name": self.name, "sensor_type": "file_watch"},
            )
        return None


class DigitalSensor(BaseSensor):
    """数字传感器——模拟/测试用，生成周期性数据."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, "digital", config)
        self._counter = 0

    async def _open(self) -> None:
        pass

    async def _close(self) -> None:
        pass

    async def _read(self) -> PerceptionEvent | None:
        self._counter += 1
        import random
        return PerceptionEvent(
            source="digital",
            event_type="data_change",
            payload={
                "counter": self._counter,
                "value": round(random.uniform(0, 100), 2),
                "sensor": self.name,
            },
            metadata={"sensor_name": self.name, "sensor_type": "digital"},
        )


class HeartbeatSensor(BaseSensor):
    """心跳传感器——定期发送系统心跳."""

    def __init__(self, name: str = "heartbeat", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, "system", config)

    async def _open(self) -> None:
        pass

    async def _close(self) -> None:
        pass

    async def _read(self) -> PerceptionEvent | None:
        import psutil
        return PerceptionEvent(
            source="system",
            event_type="heartbeat",
            payload={
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            },
            metadata={"sensor_name": self.name, "sensor_type": "heartbeat"},
        )
