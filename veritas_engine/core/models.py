"""Core Pydantic models shared across all layers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ───────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────

class EventType(str, Enum):
    """感知事件类型."""
    DATA_CHANGE = "data_change"
    ANOMALY = "anomaly"
    HEARTBEAT = "heartbeat"
    HUMAN_COMMAND = "human_command"
    STRATEGY_RESULT = "strategy_result"
    AUDIT_TRIGGER = "audit_trigger"
    SYSTEM_STATUS = "system_status"
    REASONING_START = "reasoning_start"
    REASONING_COMPLETE = "reasoning_complete"
    EMOTION_UPDATE = "emotion_update"
    KNOWLEDGE_CREATED = "knowledge_created"
    GOAL_CHANGED = "goal_changed"


class EntityType(str, Enum):
    """知识图谱实体类型."""
    DEVICE = "device"
    SQL_QUERY = "sql_query"
    REPORT = "report"
    CONCEPT = "concept"
    PROVEN_STRATEGY = "proven_strategy"
    META_LESSON = "meta_lesson"
    OBSERVATION = "observation"


class StrategySource(str, Enum):
    """策略来源."""
    SELF_PLAY = "self_play"
    EXHAUSTIVE = "exhaustive"
    PATTERN_MINING = "pattern_mining"
    CAUSAL_INFERENCE = "causal_inference"
    HUMAN = "human"


class StrategyStatus(str, Enum):
    """策略执行状态."""
    PENDING = "pending"
    SANDBOX_TESTING = "sandbox_testing"
    AWAITING_HITL = "awaiting_hitl"
    GRAY_EXECUTING = "gray_executing"
    EXECUTED = "executed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class LayerName(str, Enum):
    """系统层级名称."""
    SENSORIUM = "sensorium"
    NOOSPHERE = "noosphere"
    HYPERION = "hyperion"
    DAEMON = "daemon"
    PROMETHEUS = "prometheus"
    ORACLE = "oracle"


# ───────────────────────────────────────────────
# Core Models
# ───────────────────────────────────────────────

class PerceptionEvent(BaseModel):
    """标准化感知事件——所有外部信号的通用格式.

    这是整个系统的数据入口。无论信号来自工业传感器、数据库、
    文件系统还是人类操作员，都会被封装为 PerceptionEvent。
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # 信号来源: opcua / mssql / nodered / file / human / api
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType = EventType.DATA_CHANGE
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    def to_nats_subject(self) -> str:
        """转换为NATS subject格式."""
        return f"veritas.{self.source}.{self.event_type.value}"


class Entity(BaseModel):
    """知识图谱中的实体节点."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EntityType
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    def to_cypher_properties(self) -> dict[str, Any]:
        """转换为Cypher查询的属性字典."""
        import json
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "attributes": json.dumps(self.attributes, ensure_ascii=False),
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
        }


class Observation(BaseModel):
    """原始观测数据——存储在向量记忆中的非结构化经验."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_data: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = None  # 由LLM生成
    metadata: dict[str, Any] = Field(default_factory=dict)


class Strategy(BaseModel):
    """推演层生成的候选策略.

    策略经过情感层评估后，由执行层尝试执行。
    整个生命周期: pending → sandbox_testing → awaiting_hitl → gray_executing → executed
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source: StrategySource = StrategySource.SELF_PLAY
    sandbox_result: dict[str, Any] | None = None
    status: StrategyStatus = StrategyStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: datetime | None = None
    executed_at: datetime | None = None
    knowledge_graph_entity_id: str | None = None  # 固化后的实体ID

    def to_knowledge_entity(self) -> Entity:
        """转换为知识图谱实体（用于固化）."""
        return Entity(
            type=EntityType.PROVEN_STRATEGY,
            name=self.name,
            attributes={
                "description": self.description,
                "actions": self.actions,
                "expected_outcome": self.expected_outcome,
                "confidence": self.confidence,
                "source": self.source.value,
                "sandbox_result": self.sandbox_result,
            },
            confidence=self.confidence,
        )


class EmotionalState(BaseModel):
    """情感层当前状态——内在奖励机制的数字表达.

    不是模拟人类情绪，而是强化学习的内在奖励源。
    直接修改推演层的目标函数和探索-利用平衡参数。
    """

    curiosity: float = 0.5      # 好奇心: reward = -log(p)
    urgency: float = 0.3        # 紧迫感: discount = e^(-λt)
    frustration: float = 0.0    # 挫败感: penalty = -|expected - actual|
    achievement: float = 0.0    # 成就感: bonus = Σ(confirmed_knowledge)
    value_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "efficiency": 0.3,
            "cost": 0.25,
            "risk": 0.25,
            "innovation": 0.2,
        }
    )

    # ── 派生属性 ──

    @property
    def exploration_bias(self) -> float:
        """探索偏向 (好奇心驱动) → 增加 ε-greedy 的 ε."""
        return self.curiosity * 0.7 + self.achievement * 0.3

    @property
    def exploitation_bias(self) -> float:
        """利用偏向 (紧迫感驱动) → 减少搜索深度，优先已知路径."""
        return self.urgency * 0.6 + (1 - self.frustration) * 0.4

    @property
    def epsilon(self) -> float:
        """计算当前的 ε-greedy 探索率.

        当系统长期没有新发现时，好奇心自动提升 ε，增加随机探索。
        当 deadline 临近，紧迫感降低 ε，优先使用已知最优路径。
        """
        base = 0.1
        curiosity_boost = self.curiosity * 0.3
        urgency_reduction = -self.urgency * 0.2
        frustration_boost = self.frustration * 0.1
        return max(0.01, min(0.5, base + curiosity_boost + urgency_reduction + frustration_boost))

    @property
    def value_function_str(self) -> str:
        """价值函数的数学表达式."""
        parts = []
        for k, w in self.value_weights.items():
            parts.append(f"{w:.2f}×{k}")
        return "V = " + " + ".join(parts)

    def evaluate_value(self, metrics: dict[str, float]) -> float:
        """计算多目标帕累托价值.

        Args:
            metrics: 各维度得分，如 {"efficiency": 0.8, "cost": 0.6, ...}

        Returns:
            加权总价值 0-1
        """
        total = 0.0
        weight_sum = 0.0
        for key, weight in self.value_weights.items():
            total += weight * metrics.get(key, 0.0)
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def compute_curiosity_reward(self, probability: float) -> float:
        """好奇心奖励: 对低概率事件给予高奖励.

        reward = -log(p)
        """
        import math
        p = max(probability, 1e-10)
        return -math.log10(p)

    def compute_urgency_discount(self, time_remaining: float, lambda_param: float = 0.1) -> float:
        """紧迫感折扣: 截止时间越近，未来奖励折扣越大.

        discount = e^(-λt)
        """
        import math
        return math.exp(-lambda_param * time_remaining)

    def compute_frustration_penalty(self, expected: float, actual: float) -> float:
        """挫败感惩罚: 验证失败时惩罚.

        penalty = -|expected - actual|
        """
        return -abs(expected - actual)

    def compute_achievement_bonus(self, confirmed_knowledge_count: int) -> float:
        """成就感奖励: 每次成功固化知识，全局奖励增加.

        bonus = Σ(confirmed_knowledge) × 0.1
        """
        return confirmed_knowledge_count * 0.1


class AuditReport(BaseModel):
    """元认知层生成的反思审计报告."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    period_start: datetime
    period_end: datetime
    tasks_reviewed: int
    invalid_explorations: list[dict[str, Any]] = Field(default_factory=list)
    biases_detected: list[dict[str, Any]] = Field(default_factory=list)
    improvements: list[dict[str, Any]] = Field(default_factory=list)
    value_weight_adjustments: dict[str, float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemStatus(BaseModel):
    """系统整体状态快照."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    layer_status: dict[LayerName, str] = Field(default_factory=dict)
    emotional_state: EmotionalState | None = None
    knowledge_count: int = 0
    pending_strategies: int = 0
    executed_strategies: int = 0
    recent_events: list[PerceptionEvent] = Field(default_factory=list)


# ───────────────────────────────────────────────
# Response Models
# ───────────────────────────────────────────────

class HealthCheck(BaseModel):
    """健康检查响应."""
    status: Literal["healthy", "degraded", "unhealthy"]
    layer: str
    details: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """通用查询结果."""
    query: str
    results: list[dict[str, Any]]
    execution_time_ms: float
    total_count: int
