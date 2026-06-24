"""Daemon — emotional layer: value function, curiosity, urgency, frustration, achievement."""

from __future__ import annotations

import math
from typing import Any

from veritas_engine.core.models import EmotionalState
from veritas_engine.core.config import get_config
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.daemon")


class Daemon:
    """情感层——内在奖励机制.

    不是模拟人类情绪，而是强化学习的内在奖励源。
    直接修改推演层的目标函数和探索-利用平衡参数。
    """

    def __init__(self) -> None:
        self.config = get_config().daemon
        self.state = EmotionalState(value_weights=self.config.default_weights.copy())
        self._confirmed_knowledge_count = 0

    def compute_curiosity_reward(self, probability: float) -> float:
        """好奇心奖励: reward = -log(p).

        对低概率事件给予高奖励，驱动探索未知。
        """
        p = max(probability, 1e-10)
        reward = -math.log10(p)
        self.state.curiosity = min(1.0, self.state.curiosity + reward * 0.01)
        return reward

    def compute_urgency_discount(self, time_remaining: float, lambda_param: float | None = None) -> float:
        """紧迫感折扣: discount = e^(-λt).

        截止时间越近，未来奖励折扣越大。
        """
        lam = lambda_param or self.config.urgency_lambda
        discount = math.exp(-lam * time_remaining)
        self.state.urgency = min(1.0, self.state.urgency + (1 - discount) * 0.05)
        return discount

    def compute_frustration_penalty(self, expected: float, actual: float) -> float:
        """挫败感惩罚: penalty = -|expected - actual|.

        验证失败时惩罚，降低对相似策略的信任。
        """
        penalty = -abs(expected - actual)
        self.state.frustration = min(1.0, max(0.0, self.state.frustration + abs(penalty) * 0.1))
        return penalty

    def compute_achievement_bonus(self, confirmed_knowledge_count: int | None = None) -> float:
        """成就感奖励: bonus = Σ(confirmed_knowledge) × 0.1.

        每次成功固化知识，全局奖励增加。
        """
        count = confirmed_knowledge_count or self._confirmed_knowledge_count
        bonus = count * 0.1
        self.state.achievement = min(1.0, bonus)
        self._confirmed_knowledge_count = count
        return bonus

    def evaluate_value(self, metrics: dict[str, float]) -> float:
        """价值函数: V = w1×efficiency + w2×cost + w3×risk + w4×innovation.

        Args:
            metrics: 各维度得分
        """
        return self.state.evaluate_value(metrics)

    def get_state(self) -> EmotionalState:
        """获取当前情感状态."""
        return self.state

    def adjust_weights(self, feedback: dict[str, float]) -> None:
        """根据外部反馈调整价值权重.

        Args:
            feedback: 各维度的调整量，如 {"efficiency": 0.1, "cost": -0.05}
        """
        for key, delta in feedback.items():
            if key in self.state.value_weights:
                self.state.value_weights[key] = max(0.0, min(1.0, self.state.value_weights[key] + delta))

        # 归一化
        total = sum(self.state.value_weights.values())
        if total > 0:
            for key in self.state.value_weights:
                self.state.value_weights[key] /= total

        logger.info("Value weights adjusted: %s", self.state.value_weights, extra={"layer": "daemon"})

    def decay(self) -> None:
        """情感衰减——模拟时间流逝对情感的影响."""
        self.state.curiosity *= self.config.curiosity_decay
        self.state.urgency *= 0.95
        self.state.frustration *= 0.9
        self.state.achievement *= 0.98

    def select_strategy(self, strategies: list[Any]) -> Any | None:
        """根据情感状态选择最优策略.

        使用 ε-greedy 结合情感偏向。
        """
        import random

        if not strategies:
            return None

        epsilon = self.state.epsilon
        if random.random() < epsilon:
            # 探索：随机选择
            return random.choice(strategies)

        # 利用：选择预期价值最高的
        best = None
        best_value = -float("inf")
        for s in strategies:
            expected = s.expected_outcome if hasattr(s, "expected_outcome") else {}
            if isinstance(expected, dict):
                value = self.evaluate_value(expected)
            else:
                value = float(expected) if expected else 0.0
            if value > best_value:
                best_value = value
                best = s

        return best
