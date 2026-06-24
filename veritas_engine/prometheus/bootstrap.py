"""Prometheus — execution layer: MCP gateway, HITL, sandbox executor, pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from veritas_engine.core.config import get_config
from veritas_engine.core.models import Strategy, StrategyStatus, Entity, EntityType
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import HITLError, MCPError, RollbackError

logger = get_logger("veritas.prometheus")


class MCPGateway:
    """MCP Server 网关——统一管理外部工具调用."""

    def __init__(self) -> None:
        self.config = get_config().prometheus
        self._servers: dict[str, Any] = {}

    def register_server(self, name: str, client: Any) -> None:
        """注册 MCP Server."""
        self._servers[name] = client
        logger.info("MCP Server [%s] registered", name, extra={"layer": "prometheus"})

    async def call(self, server: str, tool: str, params: dict[str, Any]) -> Any:
        """调用 MCP Server 工具."""
        if server not in self._servers:
            raise MCPError(server, f"Server not registered")
        client = self._servers[server]
        try:
            result = await asyncio.wait_for(
                client.call(tool, params),
                timeout=get_config().hyperion.sandbox_timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise MCPError(server, "Timeout")
        except Exception as e:
            raise MCPError(server, str(e))


class HITLManager:
    """人机确认管理器——关键操作需要人类确认."""

    def __init__(self) -> None:
        self.config = get_config().prometheus
        self._pending: dict[str, dict[str, Any]] = {}

    async def request_confirmation(self, strategy: Strategy) -> bool:
        """请求人工确认.

        Returns:
            True if approved, False if rejected or timeout.
        """
        strategy_id = strategy.id
        self._pending[strategy_id] = {
            "strategy": strategy,
            "requested_at": datetime.now(timezone.utc),
            "status": "pending",
        }

        logger.info(
            "HITL confirmation requested for strategy [%s]: %s",
            strategy_id,
            strategy.name,
            extra={"layer": "prometheus"},
        )

        # 模拟等待确认（实际应发送消息到飞书/Slack等）
        timeout = self.config.hitl_timeout_minutes * 60
        try:
            await asyncio.wait_for(self._wait_for_response(strategy_id), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending[strategy_id]["status"] = "timeout"
            logger.warning("HITL timeout for strategy [%s]", strategy_id, extra={"layer": "prometheus"})
            return False

        return self._pending[strategy_id]["status"] == "approved"

    async def _wait_for_response(self, strategy_id: str) -> None:
        """等待人类响应."""
        while self._pending.get(strategy_id, {}).get("status") == "pending":
            await asyncio.sleep(1)

    def approve(self, strategy_id: str) -> None:
        """人工批准策略."""
        if strategy_id in self._pending:
            self._pending[strategy_id]["status"] = "approved"
            logger.info("Strategy [%s] approved", strategy_id, extra={"layer": "prometheus"})

    def reject(self, strategy_id: str) -> None:
        """人工拒绝策略."""
        if strategy_id in self._pending:
            self._pending[strategy_id]["status"] = "rejected"
            logger.info("Strategy [%s] rejected", strategy_id, extra={"layer": "prometheus"})


class ExecutionPipeline:
    """试错-验证-固化流水线.

    策略生命周期:
    pending -> sandbox_testing -> awaiting_hitl -> gray_executing -> executed
                              -> rejected
                              -> rolled_back
    """

    def __init__(self) -> None:
        self.config = get_config().prometheus
        self.hitl = HITLManager()
        self.mcp = MCPGateway()

    async def execute(
        self,
        strategy: Strategy,
        sandbox_executor: Any,
        require_hitl: bool = True,
        gray_scale: float | None = None,
    ) -> dict[str, Any]:
        """执行完整流水线.

        Args:
            strategy: 要执行的策略
            sandbox_executor: 沙盒执行器
            require_hitl: 是否需要人工确认
            gray_scale: 灰度比例（默认从配置读取）
        """
        gray = gray_scale or self.config.default_gray_traffic

        # 1. 沙盒验证
        strategy.status = StrategyStatus.SANDBOX_TESTING
        sandbox_result = await sandbox_executor(strategy)
        strategy.sandbox_result = sandbox_result

        if not sandbox_result.get("success", False):
            strategy.status = StrategyStatus.REJECTED
            logger.error("Sandbox failed for strategy [%s]", strategy.id, extra={"layer": "prometheus"})
            return {"success": False, "stage": "sandbox", "error": sandbox_result.get("error")}

        # 2. 人工确认
        if require_hitl:
            strategy.status = StrategyStatus.AWAITING_HITL
            approved = await self.hitl.request_confirmation(strategy)
            if not approved:
                strategy.status = StrategyStatus.REJECTED
                return {"success": False, "stage": "hitl", "error": "Not approved"}

        # 3. 灰度执行
        strategy.status = StrategyStatus.GRAY_EXECUTING
        logger.info(
            "Gray executing strategy [%s] with %.0f%% traffic",
            strategy.id,
            gray * 100,
            extra={"layer": "prometheus"},
        )
        await asyncio.sleep(0.5)  # 模拟灰度执行

        # 4. 全量执行
        strategy.status = StrategyStatus.EXECUTED
        strategy.executed_at = datetime.now(timezone.utc)
        logger.info("Strategy [%s] fully executed", strategy.id, extra={"layer": "prometheus"})

        return {"success": True, "stage": "executed", "strategy_id": strategy.id}

    async def rollback(self, strategy: Strategy) -> None:
        """回滚策略."""
        strategy.status = StrategyStatus.ROLLED_BACK
        logger.warning("Strategy [%s] rolled back", strategy.id, extra={"layer": "prometheus"})

    def solidify_knowledge(self, strategy: Strategy) -> Entity:
        """将验证通过的策略固化为知识."""
        entity = strategy.to_knowledge_entity()
        logger.info(
            "Strategy [%s] solidified as knowledge entity [%s]",
            strategy.id,
            entity.id,
            extra={"layer": "prometheus"},
        )
        return entity


class Prometheus:
    """执行层统一入口."""

    def __init__(self) -> None:
        self.pipeline = ExecutionPipeline()

    async def execute_strategy(
        self,
        strategy: Strategy,
        sandbox_executor: Any,
        require_hitl: bool = True,
        gray_scale: float | None = None,
    ) -> dict[str, Any]:
        """执行策略."""
        return await self.pipeline.execute(strategy, sandbox_executor, require_hitl, gray_scale)

    async def sandbox_validate(self, strategy: Strategy, sandbox: Any) -> dict[str, Any]:
        """沙盒预验证."""
        return await sandbox.execute(strategy)

    async def request_hitl_confirmation(self, strategy: Strategy) -> bool:
        """请求人工确认."""
        return await self.pipeline.hitl.request_confirmation(strategy)

    async def gray_execute(self, strategy: Strategy, traffic_percent: float) -> dict[str, Any]:
        """灰度执行."""
        strategy.status = StrategyStatus.GRAY_EXECUTING
        logger.info("Gray execute: %.0f%% traffic", traffic_percent * 100, extra={"layer": "prometheus"})
        return {"success": True, "traffic_percent": traffic_percent}

    async def rollback(self, strategy: Strategy) -> None:
        """回滚策略."""
        await self.pipeline.rollback(strategy)

    def solidify_knowledge(self, strategy: Strategy) -> Entity:
        """固化知识."""
        return self.pipeline.solidify_knowledge(strategy)
