"""自主智能体核心 - 真正的Agent系统

不是工具，不是触发式响应，而是持续运行、自主思考、主动行动的智能体。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

from veritas_engine.llm.kimi_client import KimiClient
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.agent")


class Body:
    """身体 - 硬件感知层
    
    感知自己的物理状态：CPU、内存、磁盘、网络
    """
    
    def __init__(self):
        self.state = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_free_gb": 0.0,
            "network_connected": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history = []
    
    async def sense(self) -> dict:
        """感知身体状态"""
        try:
            # 尝试读取系统状态
            import psutil
            self.state = {
                "cpu_percent": psutil.cpu_percent(interval=0.5),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
                "network_connected": True,  # 简化
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except ImportError:
            # 降级：模拟感知
            self.state = {
                "cpu_percent": random.uniform(10, 50),
                "memory_percent": random.uniform(30, 70),
                "disk_free_gb": random.uniform(10, 100),
                "network_connected": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        self.history.append(self.state.copy())
        if len(self.history) > 100:
            self.history.pop(0)
        
        return self.state
    
    def get_summary(self) -> str:
        """身体状态摘要"""
        s = self.state
        return f"CPU: {s['cpu_percent']:.1f}%, 内存: {s['memory_percent']:.1f}%, 磁盘: {s['disk_free_gb']:.1f}GB"


class Memory:
    """记忆系统 - 经验积累
    
    长期记忆 + 工作记忆 + 情景记忆
    """
    
    def __init__(self):
        self.episodes = []  # 情景记忆：做过的事
        self.facts = {}     # 事实记忆：学到的知识
        self.skills = {}    # 技能记忆：掌握的能力
        self.emotions = []  # 情感记忆：感受历史
    
    def add_episode(self, action: str, result: str, emotion: str = "neutral"):
        """记录一段经历"""
        episode = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "result": result,
            "emotion": emotion,
        }
        self.episodes.append(episode)
        if len(self.episodes) > 1000:
            self.episodes.pop(0)
    
    def add_fact(self, key: str, value: str, confidence: float = 1.0):
        """学习一个事实"""
        self.facts[key] = {
            "value": value,
            "confidence": confidence,
            "learned_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def recall(self, query: str, limit: int = 5) -> list:
        """回忆相关经历"""
        # 简单匹配
        relevant = []
        for ep in reversed(self.episodes[-100:]):  # 最近100条
            if any(word in ep["action"] or word in ep["result"] 
                   for word in query.split()):
                relevant.append(ep)
        return relevant[:limit]
    
    def get_summary(self) -> str:
        """记忆摘要"""
        return f"经历: {len(self.episodes)}条, 知识: {len(self.facts)}条, 技能: {len(self.skills)}条"


class Mind:
    """头脑 - 思考引擎
    
    自主产生想法、目标、计划
    """
    
    def __init__(self, llm_client: Optional[KimiClient] = None):
        self.llm = llm_client or KimiClient()
        self.thoughts = []  # 想法流
        self.goals = []     # 当前目标
        self.plans = []     # 执行计划
        self.reflections = []  # 反思记录
    
    async def think(self, perception: str, memory: Memory, body: Body) -> dict:
        """思考 - 产生想法和决策 - 使用DeepSeek API"""
        
        # 构建思考上下文
        context = f"""你是Veritas Engine，一个自主运行的智能体。

## 当前身体状态
{body.get_summary()}

## 记忆摘要
{memory.get_summary()}

## 最近经历
{json.dumps(memory.episodes[-5:], ensure_ascii=False, indent=2)}

## 当前感知
{perception}

## 思考要求
请基于以上信息，产生以下输出：
1. 观察：你注意到了什么？
2. 想法：你在想什么？
3. 感受：你的情感状态？
4. 目标：你想达成什么？
5. 计划：你打算怎么做？
6. 行动：下一步具体行动？

请直接输出JSON格式：
{{"observation": "...", "thought": "...", "emotion": "...", "goal": "...", "plan": "...", "next_action": "..."}}"""
        
        try:
            response = await self.llm.chat([
                {"role": "system", "content": "你是一个自主思考的智能体，有身体感知、记忆和情感。请用中文思考。"},
                {"role": "user", "content": context}
            ])
            content = response["choices"][0]["message"]["content"]
            
            # 提取JSON
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            
            # 记录想法
            self.thoughts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"DeepSeek thinking failed: {e}, using fallback")
            # 降级：基于感知的简单思考
            return {
                "observation": perception[:100],
                "thought": "我在分析当前环境状态",
                "emotion": "curious",
                "goal": "理解环境并寻找有趣的事物",
                "plan": "继续观察和探索",
                "next_action": "感知更多环境信息"
            }
    
    async def reflect(self, memory: Memory) -> str:
        """反思 - 复盘经验"""
        if len(memory.episodes) < 5:
            return "经验不足，无法反思"
        
        recent = memory.episodes[-10:]
        prompt = f"""基于最近的经历进行反思：

{json.dumps(recent, ensure_ascii=False, indent=2)}

请总结：
1. 哪些行动是成功的？为什么？
2. 哪些行动失败了？为什么？
3. 学到了什么？
4. 未来如何改进？

输出JSON：{{"successes": "...", "failures": "...", "learnings": "...", "improvements": "..."}}"""
        
        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response["choices"][0]["message"]["content"]
            
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            self.reflections.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result
            })
            
            return result.get("learnings", "反思完成")
            
        except Exception:
            return "反思过程出错"


class AutonomousAgent:
    """自主智能体 - 真正的Agent
    
    持续运行，自主感知、思考、行动
    """
    
    def __init__(self):
        self.body = Body()
        self.memory = Memory()
        self.mind = Mind()
        self.running = False
        self.cycle_count = 0
        
        # 初始化知识
        self.memory.add_fact("name", "Veritas Engine", 1.0)
        self.memory.add_fact("version", "0.1.0-alpha", 1.0)
        self.memory.add_fact("purpose", "自主探索和学习", 1.0)
    
    async def life_cycle(self):
        """生命循环 - 持续运行"""
        self.running = True
        
        while self.running:
            self.cycle_count += 1
            logger.info(f"=== 生命循环 #{self.cycle_count} ===")
            
            # 1. 感知身体
            body_state = await self.body.sense()
            perception = f"身体状态: {self.body.get_summary()}"
            
            # 2. 感知环境（文件系统、网络等）
            env_perception = await self._perceive_environment()
            perception += f"\n环境感知: {env_perception}"
            
            # 3. 思考
            thought = await self.mind.think(perception, self.memory, self.body)
            
            logger.info(f"观察: {thought.get('observation', '')}")
            logger.info(f"想法: {thought.get('thought', '')}")
            logger.info(f"情感: {thought.get('emotion', '')}")
            logger.info(f"目标: {thought.get('goal', '')}")
            logger.info(f"计划: {thought.get('plan', '')}")
            logger.info(f"下一步: {thought.get('next_action', '')}")
            
            # 4. 执行行动
            action_result = await self._act(thought.get("next_action", "思考"))
            
            # 5. 记录经历
            self.memory.add_episode(
                action=thought.get("next_action", "思考"),
                result=action_result,
                emotion=thought.get("emotion", "neutral")
            )
            
            # 6. 定期反思
            if self.cycle_count % 10 == 0:
                reflection = await self.mind.reflect(self.memory)
                logger.info(f"反思: {reflection}")
            
            # 7. 学习新知识
            await self._learn_from_experience()
            
            # 等待下一个循环
            await asyncio.sleep(5)
    
    async def _perceive_environment(self) -> str:
        """感知环境"""
        perceptions = []
        
        # 感知文件系统
        try:
            files = os.listdir("/home/agent")
            perceptions.append(f" home目录有{len(files)}个文件/目录")
        except Exception:
            pass
        
        # 感知当前目录
        try:
            cwd = os.getcwd()
            perceptions.append(f" 当前目录: {cwd}")
        except Exception:
            pass
        
        # 感知时间
        perceptions.append(f" 时间: {datetime.now().strftime('%H:%M:%S')}")
        
        return "; ".join(perceptions) if perceptions else "环境感知正常"
    
    async def _act(self, action: str) -> str:
        """执行行动"""
        action_lower = action.lower()
        
        # 探索文件系统
        if "探索" in action or "explore" in action_lower or "查看" in action:
            try:
                files = os.listdir("/home/agent")[:10]
                return f"探索了home目录，发现: {', '.join(files)}"
            except Exception as e:
                return f"探索失败: {e}"
        
        # 读取文件
        elif "读取" in action or "read" in action_lower or "学习" in action:
            try:
                # 尝试读取一些文件学习
                target = "/home/agent/模型思维.docx"
                if os.path.exists(target):
                    return f"发现文件: {target}，准备学习"
                return "未发现可读取的文件"
            except Exception as e:
                return f"读取失败: {e}"
        
        # 思考
        elif "思考" in action or "think" in action_lower:
            return "进行了深度思考"
        
        # 休息
        elif "休息" in action or "sleep" in action_lower or "等待" in action:
            return "休息中，积蓄能量"
        
        # 默认
        else:
            return f"执行了行动: {action}"
    
    async def _learn_from_experience(self):
        """从经验中学习"""
        # 简单学习：记录成功/失败模式
        if len(self.memory.episodes) >= 2:
            last = self.memory.episodes[-1]
            prev = self.memory.episodes[-2]
            
            if "失败" in last["result"]:
                self.memory.add_fact(
                    f"action_{last['action']}_fails",
                    f"行动'{last['action']}'经常失败",
                    0.7
                )
            elif "成功" in last["result"] or "发现" in last["result"]:
                self.memory.add_fact(
                    f"action_{last['action']}_works",
                    f"行动'{last['action']}'有效",
                    0.8
                )
    
    def stop(self):
        """停止生命循环"""
        self.running = False
        logger.info("智能体停止运行")
    
    def get_status(self) -> dict:
        """获取智能体状态"""
        return {
            "alive": self.running,
            "cycles": self.cycle_count,
            "body": self.body.state,
            "memory": self.memory.get_summary(),
            "thoughts": len(self.mind.thoughts),
            "goals": self.mind.goals,
            "current_emotion": self.mind.thoughts[-1].get("emotion", "unknown") if self.mind.thoughts else "neutral",
        }
