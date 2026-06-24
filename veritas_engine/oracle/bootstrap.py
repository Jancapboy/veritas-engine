"""Oracle — metacognitive layer: goal decomposition, reflection audit, direction control."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

from veritas_engine.core.config import get_config
from veritas_engine.core.models import AuditReport, EmotionalState, Strategy
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import GoalDecompositionError

logger = get_logger("veritas.oracle")


class GoalDecomposer:
    """目标分解器——将高层目标分解为子任务 DAG."""

    async def decompose(self, high_level_goal: str) -> list[dict[str, Any]]:
        """分解目标为子任务.

        Returns:
            子任务列表，每项包含 id, name, dependencies, action
        """
        # 简单启发式分解（实际应使用 LLM 进行 HTN 分解）
        goal_lower = high_level_goal.lower()

        tasks = []
        if "优化" in goal_lower or "optimize" in goal_lower:
            tasks = [
                {"id": "t1", "name": "数据收集与分析", "dependencies": [], "action": "collect_data"},
                {"id": "t2", "name": "瓶颈识别", "dependencies": ["t1"], "action": "identify_bottlenecks"},
                {"id": "t3", "name": "策略推演", "dependencies": ["t2"], "action": "generate_strategies"},
                {"id": "t4", "name": "沙盒验证", "dependencies": ["t3"], "action": "sandbox_test"},
                {"id": "t5", "name": "执行与监控", "dependencies": ["t4"], "action": "execute_and_monitor"},
            ]
        elif "发现" in goal_lower or "discover" in goal_lower:
            tasks = [
                {"id": "t1", "name": "数据探索", "dependencies": [], "action": "explore_data"},
                {"id": "t2", "name": "模式挖掘", "dependencies": ["t1"], "action": "mine_patterns"},
                {"id": "t3", "name": "因果验证", "dependencies": ["t2"], "action": "causal_verify"},
                {"id": "t4", "name": "知识固化", "dependencies": ["t3"], "action": "solidify_knowledge"},
            ]
        else:
            tasks = [
                {"id": "t1", "name": "理解目标", "dependencies": [], "action": "understand_goal"},
                {"id": "t2", "name": "制定计划", "dependencies": ["t1"], "action": "create_plan"},
                {"id": "t3", "name": "执行计划", "dependencies": ["t2"], "action": "execute_plan"},
                {"id": "t4", "name": "验证结果", "dependencies": ["t3"], "action": "verify_result"},
            ]

        logger.info("Goal decomposed into %d tasks", len(tasks), extra={"layer": "oracle"})
        return tasks


class Reflector:
    """反思审计器——定期认知审计."""

    def __init__(self) -> None:
        self.config = get_config().oracle

    async def conduct_audit(
        self,
        strategies: list[Strategy],
        emotional_state: EmotionalState,
        working_memory_summary: dict[str, Any],
    ) -> AuditReport:
        """执行认知审计.

        Args:
            strategies: 最近执行的策略列表
            emotional_state: 当前情感状态
            working_memory_summary: 工作记忆摘要
        """
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=self.config.audit_interval_hours)

        # 分析无效探索
        invalid_explorations = []
        for s in strategies:
            if isinstance(s.status, str):
                status_val = s.status
            else:
                status_val = s.status.value
            if status_val in ["rejected", "rolled_back"]:
                invalid_explorations.append({
                    "strategy_id": s.id,
                    "name": s.name,
                    "reason": status_val,
                    "source": s.source.value if hasattr(s.source, 'value') else str(s.source),
                })

        # 检测偏见
        biases_detected = []
        weights = emotional_state.value_weights
        max_weight = max(weights.values())
        min_weight = min(weights.values())
        if max_weight > 0.5:
            biases_detected.append({
                "type": "weight_imbalance",
                "description": f"价值权重过度偏向 {[k for k, v in weights.items() if v == max_weight][0]}",
                "severity": "medium",
            })
        if emotional_state.frustration > 0.7:
            biases_detected.append({
                "type": "frustration_loop",
                "description": "挫败感过高，可能陷入重复失败循环",
                "severity": "high",
            })

        # 改进建议
        improvements = []
        if len(invalid_explorations) > self.config.audit_task_threshold:
            improvements.append({
                "type": "exploration_efficiency",
                "description": f"无效探索过多 ({len(invalid_explorations)} 次)，建议调整好奇心参数",
                "action": "increase_curiosity_decay",
            })
        if emotional_state.curiosity < 0.2:
            improvements.append({
                "type": "curiosity_boost",
                "description": "好奇心过低，建议引入新的数据源",
                "action": "add_new_sensor",
            })

        # 价值权重调整建议
        value_adjustments = None
        if biases_detected:
            value_adjustments = {k: 0.25 for k in weights.keys()}  # 建议回归均衡

        report = AuditReport(
            period_start=period_start,
            period_end=now,
            tasks_reviewed=len(strategies),
            invalid_explorations=invalid_explorations,
            biases_detected=biases_detected,
            improvements=improvements,
            value_weight_adjustments=value_adjustments,
        )

        logger.info("Audit completed: %d tasks reviewed, %d biases detected", len(strategies), len(biases_detected), extra={"layer": "oracle"})
        return report


class DirectionController:
    """方向控制器——根据外部反馈调整系统方向."""

    def __init__(self) -> None:
        self._current_direction: str = "exploration"
        self._history: list[dict[str, Any]] = []

    def adjust(self, feedback: dict[str, Any]) -> dict[str, float]:
        """根据反馈调整方向.

        Returns:
            价值权重调整量
        """
        adjustments = {}

        if feedback.get("success_rate", 1.0) < 0.3:
            # 成功率低，增加风险规避
            adjustments["risk"] = 0.1
            self._current_direction = "cautious"
        elif feedback.get("novelty", 0.0) < 0.1:
            # 新发现少，增加创新权重
            adjustments["innovation"] = 0.1
            self._current_direction = "innovation"
        elif feedback.get("efficiency", 1.0) < 0.5:
            # 效率低，增加效率权重
            adjustments["efficiency"] = 0.1
            self._current_direction = "efficiency"
        else:
            self._current_direction = "balanced"

        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feedback": feedback,
            "adjustments": adjustments,
            "direction": self._current_direction,
        })

        return adjustments

    def get_direction(self) -> str:
        """获取当前方向."""
        return self._current_direction

    def get_history(self) -> list[dict[str, Any]]:
        """获取调整历史."""
        return self._history


class Oracle:
    """元认知层统一入口."""

    def __init__(self) -> None:
        self.decomposer = GoalDecomposer()
        self.reflector = Reflector()
        self.director = DirectionController()
        self._current_plan: list[dict[str, Any]] = []

    async def decompose_goal(self, high_level_goal: str) -> list[dict[str, Any]]:
        """目标分解."""
        self._current_plan = await self.decomposer.decompose(high_level_goal)
        return self._current_plan

    async def conduct_audit(
        self,
        strategies: list[Strategy],
        emotional_state: EmotionalState,
        working_memory_summary: dict[str, Any],
    ) -> AuditReport:
        """反思审计."""
        return await self.reflector.conduct_audit(strategies, emotional_state, working_memory_summary)

    def adjust_direction(self, feedback: dict[str, Any]) -> dict[str, float]:
        """方向调整."""
        return self.director.adjust(feedback)

    def get_current_plan(self) -> list[dict[str, Any]]:
        """获取当前执行计划."""
        return self._current_plan
