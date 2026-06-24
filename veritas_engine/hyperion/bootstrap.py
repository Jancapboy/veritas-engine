"""Hyperion bootstrap — self-play engine, exhaustive search, pattern mining, sandbox."""

from __future__ import annotations

import asyncio
import copy
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from veritas_engine.core.config import get_config
from veritas_engine.core.models import Strategy, StrategySource, StrategyStatus
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import SandboxError, SandboxTimeoutError

logger = get_logger("veritas.hyperion")


@dataclass
class MCTSNode:
    """MCTS 树节点."""
    state: dict[str, Any]
    parent: MCTSNode | None = None
    action: dict[str, Any] | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[dict[str, Any]] | None = None

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def best_child(self, c_param: float = 1.414) -> MCTSNode:
        """使用 UCB1 选择最优子节点."""
        choices_weights = [
            (c.total_reward / c.visits) + c_param * math.sqrt(math.log(self.visits) / c.visits)
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, action: dict[str, Any], state: dict[str, Any]) -> MCTSNode:
        child = MCTSNode(state=state, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child


class SelfPlayEngine:
    """自博弈引擎——使用 MCTS 在状态空间中寻找最优策略."""

    def __init__(self, iterations: int = 10000, exploration_constant: float = 1.414) -> None:
        self.iterations = iterations
        self.c = exploration_constant

    async def search(
        self,
        initial_state: dict[str, Any],
        action_space: list[dict[str, Any]],
        reward_fn: Callable[[dict[str, Any]], float],
        transition_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ) -> Strategy:
        """MCTS 搜索最优策略.

        Args:
            initial_state: 初始状态
            action_space: 可用动作列表
            reward_fn: 状态评估函数
            transition_fn: 状态转移函数
        """
        root = MCTSNode(state=copy.deepcopy(initial_state), untried_actions=action_space.copy())

        for i in range(self.iterations):
            node = self._select(root)
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                state = copy.deepcopy(node.state)
                new_state = transition_fn(state, action)
                node = node.expand(action, new_state)
            reward = self._simulate(node, reward_fn, transition_fn, action_space)
            self._backpropagate(node, reward)

            if i % 1000 == 0:
                logger.debug("MCTS iteration %d", i, extra={"layer": "hyperion"})

        # 选择访问次数最多的子节点作为最优策略
        if not root.children:
            # 如果没有扩展任何子节点，随机返回一个动作
            action = random.choice(action_space) if action_space else {}
            return Strategy(
                name="mcts_strategy",
                description=f"MCTS strategy after {self.iterations} iterations (no expansion)",
                actions=[action] if action else [],
                expected_outcome={"estimated_reward": 0.0},
                confidence=0.0,
                source=StrategySource.SELF_PLAY,
            )
        best = max(root.children, key=lambda c: c.visits)
        return Strategy(
            name="mcts_strategy",
            description=f"MCTS strategy after {self.iterations} iterations",
            actions=[best.action] if best.action else [],
            expected_outcome={"estimated_reward": best.total_reward / max(best.visits, 1)},
            confidence=min(best.visits / self.iterations, 1.0),
            source=StrategySource.SELF_PLAY,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：从根节点走到叶子节点."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c)
        return node

    def _simulate(
        self,
        node: MCTSNode,
        reward_fn: Callable[[dict[str, Any]], float],
        transition_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        action_space: list[dict[str, Any]],
    ) -> float:
        """模拟阶段：从当前节点随机模拟到终止."""
        state = copy.deepcopy(node.state)
        depth = 0
        max_depth = 20
        total_reward = 0.0

        while depth < max_depth:
            action = random.choice(action_space)
            try:
                state = transition_fn(state, action)
            except Exception:
                break
            total_reward += reward_fn(state)
            depth += 1

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """反向传播：更新路径上所有节点的统计信息."""
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


class ExhaustiveExplorer:
    """穷举探索器——在离散配置空间中搜索最优组合."""

    def __init__(self, prune_rules: list[Callable[[dict], bool]] | None = None) -> None:
        self.prune_rules = prune_rules or []

    async def search(
        self,
        config_space: list[dict[str, Any]],
        evaluator: Callable[[dict[str, Any]], float],
    ) -> list[Strategy]:
        """穷举搜索配置空间.

        Args:
            config_space: 每个维度可选值的列表
            evaluator: 评估函数，返回分数
        """
        from itertools import product

        strategies = []
        total = 1
        for dim in config_space:
            total *= len(dim.get("values", [1]))

        logger.info("Exhaustive search: %d combinations", total, extra={"layer": "hyperion"})

        values_lists = [dim["values"] for dim in config_space]
        keys = [dim["name"] for dim in config_space]

        count = 0
        for combo in product(*values_lists):
            config = dict(zip(keys, combo))

            # 剪枝
            if any(rule(config) for rule in self.prune_rules):
                continue

            score = evaluator(config)
            strategies.append(
                Strategy(
                    name=f"exhaustive_{count}",
                    description=f"Config: {config}",
                    actions=[{"type": "config", "params": config}],
                    expected_outcome={"score": score},
                    confidence=min(abs(score), 1.0),
                    source=StrategySource.EXHAUSTIVE,
                )
            )
            count += 1

        # 按分数排序
        strategies.sort(key=lambda s: s.expected_outcome.get("score", 0), reverse=True)
        return strategies[:100]  # 返回 top 100


class PatternMiner:
    """数据规律挖掘器——从数据中发现可解释的规律."""

    def __init__(self, population_size: int = 1000, generations: int = 20) -> None:
        self.population_size = population_size
        self.generations = generations

    async def mine(self, data: list[dict[str, Any]], target_key: str) -> list[dict[str, Any]]:
        """从数据中发现规律.

        Args:
            data: 数据列表，每项是字典
            target_key: 目标变量名
        """
        if not data or target_key not in data[0]:
            return []

        # 纯Python统计计算，不依赖pandas/numpy
        patterns = []
        numeric_keys = []
        for key in data[0].keys():
            try:
                float(data[0][key])
                numeric_keys.append(key)
            except (TypeError, ValueError):
                pass

        # 计算相关系数
        def _mean(values):
            return sum(values) / len(values) if values else 0.0

        def _std(values):
            m = _mean(values)
            return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0

        def _corr(x, y):
            mx, my = _mean(x), _mean(y)
            sx, sy = _std(x), _std(y)
            if sx == 0 or sy == 0:
                return 0.0
            n = len(x)
            return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)

        target_values = [float(row[target_key]) for row in data]
        for feature in numeric_keys:
            if feature == target_key:
                continue
            feature_values = [float(row[feature]) for row in data]
            corr = _corr(feature_values, target_values)
            patterns.append({
                "type": "correlation",
                "feature": feature,
                "target": target_key,
                "correlation": round(corr, 4),
                "description": f"{feature} 与 {target_key} 的相关系数为 {corr:.4f}",
            })

        # 简单规则发现（基于阈值）
        for col in numeric_keys:
            if col == target_key:
                continue
            col_values = [float(row[col]) for row in data]
            mean_val = _mean(col_values)
            high_rows = [row for row, cv in zip(data, col_values) if cv > mean_val]
            low_rows = [row for row, cv in zip(data, col_values) if cv <= mean_val]
            if high_rows and low_rows:
                target_mean_high = _mean([float(row[target_key]) for row in high_rows])
                target_mean_low = _mean([float(row[target_key]) for row in low_rows])
                if abs(target_mean_high - target_mean_low) > 0.01:
                    patterns.append({
                        "type": "threshold_rule",
                        "feature": col,
                        "threshold": round(mean_val, 4),
                        "target": target_key,
                        "description": f"当 {col} > {mean_val:.4f} 时，{target_key} 平均为 {target_mean_high:.4f}，否则为 {target_mean_low:.4f}",
                    })

        return patterns


class Sandbox:
    """沙盒环境——隔离执行策略，防止破坏生产系统."""

    def __init__(self) -> None:
        self.config = get_config().hyperion

    async def execute(self, strategy: Strategy) -> dict[str, Any]:
        """在沙盒中执行策略.

        Returns:
            {"success": bool, "output": Any, "error": str | None, "duration_ms": float}
        """
        import time
        start = time.time()

        # 安全检查
        for action in strategy.actions:
            if not self._validate_action(action):
                return {
                    "success": False,
                    "output": None,
                    "error": "Action validation failed",
                    "duration_ms": (time.time() - start) * 1000,
                }

        # 模拟执行（实际应使用 subprocess + 资源限制）
        try:
            result = await asyncio.wait_for(
                self._run_actions(strategy.actions),
                timeout=self.config.sandbox_timeout,
            )
            return {
                "success": True,
                "output": result,
                "error": None,
                "duration_ms": (time.time() - start) * 1000,
            }
        except asyncio.TimeoutError:
            raise SandboxTimeoutError(self.config.sandbox_timeout)
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "duration_ms": (time.time() - start) * 1000,
            }

    def _validate_action(self, action: dict[str, Any]) -> bool:
        """验证动作是否安全."""
        action_type = action.get("type", "")
        if action_type == "sql":
            sql = action.get("params", {}).get("sql", "").upper()
            blacklist = self.config.sql_blacklist if hasattr(self.config, "sql_blacklist") else ["DROP", "DELETE", "TRUNCATE"]
            for word in blacklist:
                if word in sql:
                    return False
        return True

    async def _run_actions(self, actions: list[dict[str, Any]]) -> Any:
        """模拟执行动作序列."""
        results = []
        for action in actions:
            await asyncio.sleep(0.01)  # 模拟执行时间
            results.append({"action": action, "status": "simulated_ok"})
        return results


class Hyperion:
    """推演层统一入口."""

    def __init__(self) -> None:
        self.config = get_config().hyperion
        self.self_play = SelfPlayEngine(
            iterations=self.config.mcts_iterations,
            exploration_constant=self.config.mcts_exploration_constant,
        )
        self.exhaustive = ExhaustiveExplorer()
        self.miner = PatternMiner(
            population_size=self.config.pattern_miner_population_size,
            generations=self.config.pattern_miner_generations,
        )
        self.sandbox = Sandbox()

    async def self_play_search(
        self,
        initial_state: dict[str, Any],
        action_space: list[dict[str, Any]],
        reward_fn: Callable[[dict[str, Any]], float],
        transition_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ) -> Strategy:
        """自博弈搜索."""
        return await self.self_play.search(initial_state, action_space, reward_fn, transition_fn)

    async def exhaustive_search(
        self,
        config_space: list[dict[str, Any]],
        evaluator: Callable[[dict[str, Any]], float],
    ) -> list[Strategy]:
        """穷举搜索."""
        return await self.exhaustive.search(config_space, evaluator)

    async def mine_patterns(self, data: list[dict[str, Any]], target_key: str) -> list[dict[str, Any]]:
        """规律挖掘."""
        return await self.miner.mine(data, target_key)

    async def sandbox_test(self, strategy: Strategy) -> dict[str, Any]:
        """沙盒测试策略."""
        return await self.sandbox.execute(strategy)
