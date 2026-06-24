"""内置思维模型库 - 基于《模型思维》"""

from __future__ import annotations

import math
import random
from typing import Any


class BaseModel:
    """所有思维模型的基类"""
    
    def apply(self, problem: str, context: dict) -> dict:
        """应用模型分析问题"""
        raise NotImplementedError


class NormalDistribution(BaseModel):
    """正态分布模型 - 分析均值、方差、极端值"""
    
    def apply(self, problem: str, context: dict) -> dict:
        data = context.get("data", [])
        if not data:
            return {"conclusion": "无数据，无法应用正态分布分析"}
        
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(variance)
        
        # 计算极端值概率 (>2σ)
        outliers = [x for x in data if abs(x - mean) > 2 * std]
        outlier_prob = len(outliers) / n
        
        return {
            "model": "正态分布",
            "mean": round(mean, 2),
            "std": round(std, 2),
            "variance": round(variance, 2),
            "outlier_probability": round(outlier_prob, 2),
            "conclusion": f"数据均值{mean:.1f}，标准差{std:.1f}。{'存在' if outlier_prob > 0.05 else '无明显'}极端值。",
            "insight": "若数据近似正态，约68%在1σ内，95%在2σ内。"
        }


class PowerLaw(BaseModel):
    """幂律分布模型 - 分析长尾、头部效应"""
    
    def apply(self, problem: str, context: dict) -> dict:
        data = context.get("data", [])
        if not data:
            return {"conclusion": "无数据"}
        
        sorted_data = sorted(data, reverse=True)
        n = len(sorted_data)
        
        # 80-20 分析
        total = sum(sorted_data)
        cumsum = 0
        top20_idx = int(n * 0.2)
        top20_sum = sum(sorted_data[:top20_idx])
        top20_ratio = top20_sum / total if total > 0 else 0
        
        return {
            "model": "幂律分布",
            "top_20_percent_contribution": round(top20_ratio, 2),
            "conclusion": f"前20%贡献了{top20_ratio:.1%}的总量。{'符合' if top20_ratio > 0.7 else '不符合'}典型的80-20法则。",
            "insight": "幂律分布意味着少数关键因素产生大部分影响，应优先优化头部。"
        }


class LinearModel(BaseModel):
    """线性模型 - 因果关系、趋势分析"""
    
    def apply(self, problem: str, context: dict) -> dict:
        x = context.get("x", [])
        y = context.get("y", [])
        
        if not x or not y or len(x) != len(y):
            return {"conclusion": "需要成对的x,y数据"}
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # 最小二乘法
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)
        
        if denominator == 0:
            return {"conclusion": "x无变化，无法拟合"}
        
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # R²
        ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "model": "线性模型",
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "conclusion": f"y = {slope:.4f}x + {intercept:.4f}，R²={r_squared:.4f}。{'强相关' if r_squared > 0.7 else '中等相关' if r_squared > 0.3 else '弱相关'}。",
            "insight": f"x每增加1，y平均变化{slope:.2f}。"
        }


class NetworkModel(BaseModel):
    """网络模型 - 连接关系、中心性"""
    
    def apply(self, problem: str, context: dict) -> dict:
        edges = context.get("edges", [])  # [(a,b), (b,c), ...]
        if not edges:
            return {"conclusion": "无网络连接数据"}
        
        # 构建邻接表
        nodes = set()
        adjacency = {}
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        
        # 计算度中心性
        degrees = {node: len(adjacency.get(node, [])) for node in nodes}
        max_degree_node = max(degrees, key=degrees.get)
        avg_degree = sum(degrees.values()) / len(nodes)
        
        return {
            "model": "网络模型",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "avg_degree": round(avg_degree, 2),
            "most_connected": max_degree_node,
            "conclusion": f"网络有{len(nodes)}个节点，{len(edges)}条边。平均度{avg_degree:.1f}。'{max_degree_node}'是中心节点。",
            "insight": "网络中的关键节点控制着信息/影响的流动。"
        }


class ContagionModel(BaseModel):
    """传染模型(SIR) - 扩散过程、临界点"""
    
    def apply(self, problem: str, context: dict) -> dict:
        # 基本参数
        beta = context.get("infection_rate", 0.3)  # 感染率
        gamma = context.get("recovery_rate", 0.1)  # 恢复率
        R0 = beta / gamma if gamma > 0 else 0
        
        # 模拟
        population = context.get("population", 1000)
        initial_infected = context.get("initial_infected", 1)
        steps = context.get("steps", 100)
        
        S = population - initial_infected
        I = initial_infected
        R = 0
        
        peak_infected = I
        peak_time = 0
        
        for t in range(steps):
            if S <= 0 or I <= 0:
                break
            new_infected = beta * S * I / population
            new_recovered = gamma * I
            S -= new_infected
            I += new_infected - new_recovered
            R += new_recovered
            
            if I > peak_infected:
                peak_infected = I
                peak_time = t
        
        total_infected = population - S
        
        return {
            "model": "传染模型(SIR)",
            "R0": round(R0, 2),
            "total_infected": round(total_infected, 0),
            "infection_rate": round(total_infected / population, 2),
            "peak_infected": round(peak_infected, 0),
            "peak_time": peak_time,
            "conclusion": f"基本再生数R₀={R0:.2f}。{'会大规模扩散' if R0 > 1 else '不会扩散'}。预计感染{total_infected:.0f}人({total_infected/population:.1%})。",
            "insight": f"当R₀>1时，每个感染者平均传染超过1人，疫情会指数增长。{'需要降低传播率或提高恢复率。' if R0 > 1 else '疫情会自然消退。'}"
        }


class RandomWalk(BaseModel):
    """随机游走模型 - 路径依赖、不可预测性"""
    
    def apply(self, problem: str, context: dict) -> dict:
        steps = context.get("steps", 100)
        trials = context.get("trials", 1000)
        
        # 模拟多次随机游走
        final_positions = []
        max_distances = []
        
        for _ in range(trials):
            position = 0
            max_dist = 0
            for _ in range(steps):
                position += random.choice([-1, 1])
                max_dist = max(max_dist, abs(position))
            final_positions.append(position)
            max_distances.append(max_dist)
        
        avg_final = sum(final_positions) / trials
        avg_max_dist = sum(max_distances) / trials
        
        # 理论值：E[max] ≈ sqrt(2*N/π)
        theoretical_max = math.sqrt(2 * steps / math.pi)
        
        return {
            "model": "随机游走",
            "steps": steps,
            "avg_final_position": round(avg_final, 2),
            "avg_max_distance": round(avg_max_dist, 2),
            "theoretical_max": round(theoretical_max, 2),
            "conclusion": f"{steps}步后，平均位置{avg_final:.1f}（理论上应为0）。平均最大偏离{avg_max_dist:.1f}，理论值{theoretical_max:.1f}。",
            "insight": "随机游走具有不可预测性，但长期期望位置为0。最大偏离随√N增长。"
        }


class MarkovModel(BaseModel):
    """马尔可夫模型 - 状态转移、长期均衡"""
    
    def apply(self, problem: str, context: dict) -> dict:
        # 转移矩阵
        P = context.get("transition_matrix", [[0.9, 0.1], [0.3, 0.7]])
        initial = context.get("initial_state", [1, 0])
        steps = context.get("steps", 10)
        
        state = initial[:]
        history = [state[:]]
        
        for _ in range(steps):
            new_state = [
                sum(state[i] * P[i][j] for i in range(len(P)))
                for j in range(len(P[0]))
            ]
            state = new_state
            history.append(state[:])
        
        # 计算稳态（特征向量）
        # 简化：取最后状态作为近似
        steady_state = state
        
        return {
            "model": "马尔可夫模型",
            "transition_matrix": P,
            "final_state": [round(s, 4) for s in state],
            "steady_state_approx": [round(s, 4) for s in steady_state],
            "conclusion": f"经过{steps}步转移，状态分布为{[round(s, 3) for s in state]}。",
            "insight": "马尔可夫过程具有记忆性，长期行为由转移矩阵决定，与初始状态无关。"
        }


class GameTheory(BaseModel):
    """博弈论模型 - 策略互动、纳什均衡"""
    
    def apply(self, problem: str, context: dict) -> dict:
        # 囚徒困境收益矩阵
        payoff_matrix = context.get("payoff_matrix", {
            "cooperate_cooperate": (3, 3),
            "cooperate_defect": (0, 5),
            "defect_cooperate": (5, 0),
            "defect_defect": (1, 1),
        })
        
        # 分析纳什均衡
        # 双方背叛是占优策略均衡
        cc = payoff_matrix["cooperate_cooperate"]
        cd = payoff_matrix["cooperate_defect"]
        dc = payoff_matrix["defect_cooperate"]
        dd = payoff_matrix["defect_defect"]
        
        # 检查是否双方背叛是纳什均衡
        # 给定对方背叛，我选择背叛得1，合作得0 -> 背叛更好
        # 给定对方合作，我选择背叛得5，合作得3 -> 背叛更好
        
        # 检查帕累托最优
        pareto_optimal = cc[0] + cc[1]  # 合作合作总收益
        nash_total = dd[0] + dd[1]  # 背叛背叛总收益
        
        return {
            "model": "博弈论",
            "payoff_matrix": payoff_matrix,
            "nash_equilibrium": "双方背叛",
            "pareto_optimal": "双方合作",
            "dilemma": nash_total < pareto_optimal,
            "conclusion": f"纳什均衡是双方背叛(各得{dd[0]})，但帕累托最优是双方合作(各得{cc[0]})。{'存在' if nash_total < pareto_optimal else '不存在'}囚徒困境。",
            "insight": "个体理性选择可能导致集体非最优结果。需要机制设计（如重复博弈、惩罚）来促成合作。"
        }


class LearningModel(BaseModel):
    """学习模型 - 适应过程、经验积累"""
    
    def apply(self, problem: str, context: dict) -> dict:
        # 多臂老虎机简化版
        arms = context.get("arms", 3)
        true_probs = context.get("true_probs", [0.3, 0.5, 0.7])
        trials = context.get("trials", 100)
        
        # ε-贪心策略
        epsilon = 0.1
        counts = [0] * arms
        rewards = [0.0] * arms
        total_reward = 0
        
        for t in range(trials):
            if random.random() < epsilon:
                # 探索
                arm = random.randint(0, arms - 1)
            else:
                # 利用
                avg_rewards = [rewards[i] / counts[i] if counts[i] > 0 else 0 for i in range(arms)]
                arm = avg_rewards.index(max(avg_rewards))
            
            # 拉臂
            reward = 1 if random.random() < true_probs[arm] else 0
            counts[arm] += 1
            rewards[arm] += reward
            total_reward += reward
        
        best_arm = true_probs.index(max(true_probs))
        optimal_pulls = counts[best_arm]
        
        return {
            "model": "学习模型(多臂老虎机)",
            "arms": arms,
            "trials": trials,
            "total_reward": total_reward,
            "reward_rate": round(total_reward / trials, 2),
            "optimal_arm_pulls": optimal_pulls,
            "optimal_arm_rate": round(optimal_pulls / trials, 2),
            "conclusion": f"{trials}次尝试中，总收益{total_reward}，收益率{total_reward/trials:.1%}。最优臂被选择了{optimal_pulls}次({optimal_pulls/trials:.1%})。",
            "insight": "探索-利用权衡：过多探索浪费机会，过多利用可能陷入局部最优。ε-贪心策略平衡两者。"
        }


class ThresholdModel(BaseModel):
    """阈值模型 - 临界点、级联效应"""
    
    def apply(self, problem: str, context: dict) -> dict:
        # 格兰诺维特暴动模型
        population = context.get("population", 100)
        thresholds = context.get("thresholds", [])  # 每个人的阈值
        
        if not thresholds:
            # 生成随机阈值分布
            thresholds = [random.uniform(0, 100) for _ in range(population)]
        
        thresholds.sort()
        
        # 找到均衡点
        # 如果k个人参与，阈值<=k的人会加入
        equilibrium = 0
        for k in range(1, population + 1):
            # 阈值 <= k-1 的人数（0-indexed）
            willing = sum(1 for t in thresholds if t <= k)
            if willing == k:
                equilibrium = k
                break
            elif willing < k:
                equilibrium = willing
                break
        
        # 检查是否有多个均衡
        equilibria = []
        for k in range(population + 1):
            willing = sum(1 for t in thresholds if t <= k)
            if willing == k:
                equilibria.append(k)
        
        return {
            "model": "阈值模型",
            "population": population,
            "equilibrium": equilibrium,
            "equilibria": equilibria,
            "conclusion": f"系统均衡点：{equilibrium}人参与（共{population}人）。{'存在多个均衡' if len(equilibria) > 1 else '唯一均衡'}。",
            "insight": "微小扰动可能使系统从一个均衡跳到另一个均衡。初始条件至关重要。"
        }
