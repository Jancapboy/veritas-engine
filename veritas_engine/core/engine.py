"""Veritas Engine — core bootstrap and main cognitive loop.

Wires all layers together and runs the autonomous cognitive cycle:
    Perception -> Cognition -> Oracle -> Daemon -> Hyperion -> Prometheus -> Feedback
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from typing import Any

from veritas_engine.core.config import get_config
from veritas_engine.core.models import (
    PerceptionEvent,
    Strategy,
    StrategyStatus,
    SystemStatus,
    EmotionalState,
)
from veritas_engine.core.logger import get_logger
from veritas_engine.sensorium import Sensorium
from veritas_engine.noosphere import Noosphere
from veritas_engine.hyperion import Hyperion
from veritas_engine.daemon import Daemon
from veritas_engine.prometheus import Prometheus
from veritas_engine.oracle import Oracle
from veritas_engine.llm import LLMClient, ToolRegistry

logger = get_logger("veritas.engine")


class Engine:
    """真理引擎核心——协调六层认知架构的主循环."""

    def __init__(self) -> None:
        self.config = get_config()
        self.sensorium = Sensorium()
        self.noosphere = Noosphere()
        self.hyperion = Hyperion()
        self.daemon = Daemon()
        self.prometheus = Prometheus()
        self.oracle = Oracle()
        self.llm = LLMClient()
        self.tool_registry = ToolRegistry()

        self._running = False
        self._main_task: asyncio.Task | None = None
        self._audit_task: asyncio.Task | None = None
        self._status_callbacks: list[callable] = []

        # 统计
        self._strategies: list[Strategy] = []
        self._event_count = 0

    async def start(self) -> None:
        """启动引擎——初始化所有层并启动主循环."""
        logger.info("Starting Veritas Engine v%s", self.config.version, extra={"layer": "system"})

        # 启动各层
        await self.sensorium.start()
        self.noosphere.connect()

        # 订阅感知事件 -> 认知层
        await self.sensorium.subscribe(["*"], self._on_perception)

        # 启动主循环
        self._running = True
        self._main_task = asyncio.create_task(self._cognitive_loop())
        self._audit_task = asyncio.create_task(self._audit_loop())

        logger.info("Engine started", extra={"layer": "system"})

    async def stop(self) -> None:
        """停止引擎."""
        self._running = False

        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        if self._audit_task:
            self._audit_task.cancel()
            try:
                await self._audit_task
            except asyncio.CancelledError:
                pass

        await self.sensorium.stop()
        self.noosphere.close()
        await self.llm.close()

        logger.info("Engine stopped", extra={"layer": "system"})

    async def run_goal(self, goal: str) -> dict[str, Any]:
        """执行单次目标推理.

        完整闭环:
        1. Oracle 分解目标
        2. Hyperion 推演策略
        3. Daemon 评估选择
        4. Prometheus 执行
        5. 反馈到认知层
        """
        logger.info("Running goal: %s", goal, extra={"layer": "system"})

        # 1. 目标分解
        plan = await self.oracle.decompose_goal(goal)
        logger.info("Goal decomposed into %d tasks", len(plan), extra={"layer": "system"})

        # 2. 推演策略（简化：使用穷举搜索）
        # 实际应根据目标类型选择不同推演器
        strategies: list[Strategy] = []

        # 模拟配置空间
        config_space = [
            {"name": "param_a", "values": [1, 2, 3]},
            {"name": "param_b", "values": [10, 20]},
        ]

        def _evaluator(config: dict) -> float:
            # 模拟评估：基于当前情感价值函数
            metrics = {
                "efficiency": config["param_a"] / 3.0,
                "cost": 1.0 - config["param_b"] / 30.0,
                "risk": 0.5,
                "innovation": config["param_a"] / 5.0,
            }
            return self.daemon.evaluate_value(metrics)

        strategies = await self.hyperion.exhaustive_search(config_space, _evaluator)
        logger.info("Generated %d strategies", len(strategies), extra={"layer": "system"})

        if not strategies:
            return {"success": False, "error": "No strategies generated"}

        # 3. 情感评估选择最优策略
        selected = self.daemon.select_strategy(strategies)
        if not selected:
            return {"success": False, "error": "Daemon rejected all strategies"}

        logger.info(
            "Daemon selected strategy [%s] with confidence %.2f",
            selected.id,
            selected.confidence,
            extra={"layer": "system"},
        )

        # 4. 沙盒验证 + 执行
        result = await self.prometheus.execute_strategy(
            selected,
            self.hyperion.sandbox.execute,
            require_hitl=False,  # 自动模式跳过 HITL
        )

        # 5. 记录策略和反馈
        self._strategies.append(selected)
        if result["success"]:
            self.daemon.compute_achievement_bonus(len(self._strategies))
            # 固化为知识
            entity = self.prometheus.solidify_knowledge(selected)
            self.noosphere.create_entity(entity)
            logger.info("Knowledge solidified: %s", entity.name, extra={"layer": "system"})
        else:
            self.daemon.compute_frustration_penalty(1.0, 0.0)

        return {
            "success": result["success"],
            "goal": goal,
            "strategy_id": selected.id,
            "strategy_name": selected.name,
            "stage": result.get("stage"),
            "emotional_state": self.daemon.get_state().model_dump(),
        }

    async def get_status(self) -> SystemStatus:
        """获取系统状态快照."""
        return SystemStatus(
            layer_status={
                "sensorium": "running" if self.sensorium.event_bus._running else "stopped",
                "noosphere": "connected" if self.noosphere.graph._mock is not None else "stopped",
                "hyperion": "ready",
                "daemon": "active",
                "prometheus": "ready",
                "oracle": "active",
            },
            emotional_state=self.daemon.get_state(),
            knowledge_count=len(self._strategies),
            pending_strategies=len([s for s in self._strategies if s.status == StrategyStatus.PENDING]),
            executed_strategies=len([s for s in self._strategies if s.status == StrategyStatus.EXECUTED]),
            recent_events=self.noosphere.working_memory.get_recent(5),
        )

    def register_status_callback(self, callback: callable) -> None:
        """注册状态更新回调."""
        self._status_callbacks.append(callback)

    # ── 内部循环 ──

    async def _cognitive_loop(self) -> None:
        """主认知循环——处理工作记忆中的事件."""
        while self._running:
            await asyncio.sleep(1.0)

            # 检查工作记忆
            wm = self.noosphere.get_working_memory()
            events = wm.get_recent(5)
            if not events:
                continue

            # 情感衰减
            self.daemon.decay()

            # 检查是否需要审计
            if len(self._strategies) > 0 and len(self._strategies) % 10 == 0:
                logger.debug("Strategy milestone: %d strategies", len(self._strategies), extra={"layer": "system"})

    async def _audit_loop(self) -> None:
        """定期审计循环."""
        interval = self.config.oracle.audit_interval_hours * 3600
        while self._running:
            await asyncio.sleep(min(interval, 60))  # 测试时缩短到60秒

            if not self._strategies:
                continue

            try:
                report = await self.oracle.conduct_audit(
                    self._strategies,
                    self.daemon.get_state(),
                    self.noosphere.working_memory.summary(),
                )
                logger.info(
                    "Audit: %d tasks, %d biases, %d improvements",
                    report.tasks_reviewed,
                    len(report.biases_detected),
                    len(report.improvements),
                    extra={"layer": "system"},
                )

                # 应用价值权重调整
                if report.value_weight_adjustments:
                    self.daemon.adjust_weights(report.value_weight_adjustments)

            except Exception as e:
                logger.error("Audit error: %s", e, extra={"layer": "system"})

    async def _on_perception(self, event: PerceptionEvent) -> None:
        """感知事件处理器——将事件存入认知层."""
        self._event_count += 1
        self.noosphere.add_to_working_memory(event)

        # 存储到向量记忆
        content = f"[{event.source}] {event.event_type.value}: {json.dumps(event.payload, default=str)[:200]}"
        self.noosphere.store_memory(content, metadata=event.metadata, source=event.source)

        # 好奇心奖励（新事件类型）
        self.daemon.compute_curiosity_reward(0.5)

        logger.debug(
            "Perception processed: %s from %s",
            event.event_type.value,
            event.source,
            extra={"layer": "system"},
        )

