"""多模型思维引擎 - 核心推理系统

基于《模型思维》框架，实现多模型组合推理、自我迭代优化。
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from veritas_engine.llm.kimi_client import KimiClient
from veritas_engine.models.builtin import (
    NormalDistribution, PowerLaw, LinearModel, NetworkModel,
    ContagionModel, RandomWalk, MarkovModel, GameTheory,
    LearningModel, ThresholdModel
)


class ModelThinker:
    """多模型思维引擎
    
    核心思想：用多个不同的逻辑框架"生成"智慧
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client or KimiClient()
        self.model_registry = {}
        self.reasoning_history = []
        self._register_builtin_models()
    
    def _register_builtin_models(self):
        """注册内置思维模型"""
        self.model_registry = {
            "正态分布": NormalDistribution(),
            "幂律分布": PowerLaw(),
            "线性模型": LinearModel(),
            "网络模型": NetworkModel(),
            "传染模型": ContagionModel(),
            "随机游走": RandomWalk(),
            "马尔可夫": MarkovModel(),
            "博弈论": GameTheory(),
            "学习模型": LearningModel(),
            "阈值模型": ThresholdModel(),
        }
    
    async def think(self, problem: str, context: dict = None) -> dict:
        """多模型思维主入口
        
        流程：
        1. 用LLM分析问题，选择适合的模型组合
        2. 每个模型独立推理
        3. 交叉验证，综合结论
        4. 记录推理历史，用于自我迭代
        """
        context = context or {}
        
        # 1. 问题解析 - 用LLM选择模型
        model_selection = await self._select_models(problem, context)
        
        # 2. 多模型并行推理
        results = {}
        for model_name in model_selection["models"]:
            if model_name in self.model_registry:
                model = self.model_registry[model_name]
                try:
                    result = model.apply(problem, context)
                    results[model_name] = result
                except Exception as e:
                    results[model_name] = {"error": str(e)}
        
        # 3. 交叉验证与综合
        synthesis = await self._synthesize(results, problem)
        
        # 4. 记录历史
        record = {
            "problem": problem,
            "models_used": list(results.keys()),
            "results": results,
            "synthesis": synthesis,
            "confidence": synthesis.get("confidence", 0.5),
        }
        self.reasoning_history.append(record)
        
        return {
            "answer": synthesis["answer"],
            "confidence": synthesis["confidence"],
            "models_used": list(results.keys()),
            "reasoning_chain": synthesis["reasoning"],
            "details": results,
        }
    
    async def _select_models(self, problem: str, context: dict) -> dict:
        """智能选择模型组合 - 完全基于LLM语义理解"""
        
        # 构建上下文描述
        context_desc = ""
        if context:
            if "data" in context:
                data = context["data"]
                if isinstance(data, list) and len(data) > 0:
                    context_desc += f"\n用户提供了数值数据: {data[:10]}... (共{len(data)}个数据点)"
            if "x" in context and "y" in context:
                context_desc += f"\n用户提供了成对的x,y数据"
            if "edges" in context:
                context_desc += f"\n用户提供了网络连接数据"
            if "population" in context:
                context_desc += f"\n涉及人口/群体规模: {context['population']}"
        
        prompt = f"""你是一位精通《模型思维》的专家。请分析用户的问题，选择最适合的3个思维模型组合。

## 可用模型及适用场景

1. **正态分布** - 适用于：分析数据的均值、方差、标准差；识别异常值/极端值；评估概率分布；质量控制；测量误差分析

2. **幂律分布** - 适用于：分析不平等现象（财富、收入、城市规模）；识别头部效应和长尾；80-20法则验证；排名分析；网络中的度分布

3. **线性模型** - 适用于：趋势预测和时间序列分析；因果关系推断；回归分析；斜率和变化率计算；数据拟合；预测未来值

4. **网络模型** - 适用于：分析连接关系和拓扑结构；识别关键节点和中心性；信息/疾病传播路径；社交网络分析；级联效应

5. **传染模型(SIR)** - 适用于：疫情传播模拟；病毒扩散预测；信息/谣言传播； Adoption曲线；临界点分析；干预策略评估

6. **随机游走** - 适用于：股票价格/市场波动分析；路径依赖现象；不可预测性评估；布朗运动；赌博/风险分析；长期行为预测

7. **马尔可夫模型** - 适用于：状态转移分析；长期均衡预测；记忆性过程；天气预测；客户流失分析；页面跳转分析

8. **博弈论** - 适用于：竞争策略分析；定价决策；拍卖设计；合作与背叛分析；纳什均衡；激励机制设计；多方博弈

9. **学习模型** - 适用于：多臂老虎机问题；A/B测试优化；探索-利用权衡；强化学习；自适应策略；推荐系统优化

10. **阈值模型** - 适用于：临界点分析；级联效应；社会运动/暴动预测；技术采用；意见形成；触发条件分析

## 用户问题
{problem}{context_desc}

## 输出要求
请直接输出JSON格式，不要有任何其他文字：
{{"models": ["模型名称1", "模型名称2", "模型名称3"], "reasoning": "详细解释为什么这3个模型最适合这个问题，每个模型能解决什么方面", "expected_insights": "预期能得出什么关键洞察"}}"""
        
        try:
            response = await self.llm.chat([
                {"role": "system", "content": "你是一个多模型思维专家，精通《模型思维》中的各种数学模型。你的任务是根据问题语义选择最合适的模型组合。"},
                {"role": "user", "content": prompt}
            ])
            content = response["choices"][0]["message"]["content"]
            
            # 提取JSON - 处理可能的markdown代码块
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            
            json_str = json_str.strip()
            result = json.loads(json_str)
            
            # 验证模型名称是否有效
            valid_models = list(self.model_registry.keys())
            selected = [m for m in result.get("models", []) if m in valid_models]
            
            # 如果选择的模型不足3个，补充随机选择
            if len(selected) < 3:
                remaining = [m for m in valid_models if m not in selected]
                selected.extend(random.sample(remaining, 3 - len(selected)))
            
            result["models"] = selected[:3]
            return result
            
        except Exception as e:
            # 完全降级：随机选择3个模型
            selected = random.sample(list(self.model_registry.keys()), 3)
            return {
                "models": selected,
                "reasoning": f"LLM选择失败({str(e)})，使用随机选择",
                "expected_insights": "多视角分析"
            }
    
    async def _synthesize(self, results: dict, problem: str) -> dict:
        """综合多个模型的推理结果"""
        # 构建综合提示
        results_text = json.dumps(results, ensure_ascii=False, indent=2)
        
        prompt = f"""基于以下多个思维模型的分析结果，综合得出最终结论。

原始问题：{problem}

各模型分析：
{results_text}

请：
1. 识别各模型结论的一致性和矛盾点
2. 用孔多塞陪审团定理思想，综合多数模型的共识
3. 用多样性预测定理，利用模型差异提高准确性
4. 给出置信度评估

输出JSON：
{{
    "answer": "综合结论",
    "confidence": 0.85,
    "reasoning": "推理过程",
    "disagreements": "模型间的分歧点",
    "consensus": "模型共识"
}}"""
        
        try:
            response = await self.llm.chat([
                {"role": "system", "content": "你是一个综合分析专家，擅长整合多模型结论。"},
                {"role": "user", "content": prompt}
            ])
            content = response["choices"][0]["message"]["content"]
            json_str = content[content.find("{"):content.rfind("}")+1]
            return json.loads(json_str)
        except Exception:
            # 降级：简单综合
            answers = [r.get("conclusion", "") for r in results.values() if isinstance(r, dict)]
            return {
                "answer": " | ".join(answers) if answers else "无法得出结论",
                "confidence": 0.5,
                "reasoning": "综合失败，使用原始拼接",
                "disagreements": "",
                "consensus": ""
            }
    
    def self_improve(self) -> dict:
        """自我迭代：分析历史推理，优化模型选择策略"""
        if len(self.reasoning_history) < 5:
            return {"status": "insufficient_data", "message": "需要至少5条历史记录"}
        
        # 分析哪些模型组合效果最好
        model_performance = {}
        for record in self.reasoning_history:
            models = tuple(sorted(record["models_used"]))
            conf = record["confidence"]
            if models not in model_performance:
                model_performance[models] = []
            model_performance[models].append(conf)
        
        # 计算平均置信度
        best_combinations = sorted(
            [(models, sum(confs)/len(confs)) for models, confs in model_performance.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "status": "improved",
            "best_combinations": best_combinations[:3],
            "total_records": len(self.reasoning_history),
            "insight": "已识别最优模型组合策略"
        }
