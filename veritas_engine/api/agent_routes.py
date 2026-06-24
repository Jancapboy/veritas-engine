"""自主智能体API - 与Agent交互的接口

不是工具调用，而是与一个有生命的Agent对话
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from veritas_engine.agent.autonomous import AutonomousAgent

router = APIRouter(prefix="/agent", tags=["agent"])

# 全局智能体实例
_agent: AutonomousAgent | None = None


@router.post("/spawn")
async def spawn_agent() -> dict[str, Any]:
    """孵化智能体 - 赋予生命"""
    global _agent
    
    if _agent and _agent.running:
        return {"status": "already_alive", "message": "智能体已经在运行"}
    
    _agent = AutonomousAgent()
    
    # 在后台启动生命循环
    asyncio.create_task(_agent.life_cycle())
    
    return {
        "status": "born",
        "message": "智能体已孵化，开始自主运行",
        "agent": {
            "name": "Veritas",
            "version": "0.1.0",
            "state": "alive",
        }
    }


@router.get("/status")
async def agent_status() -> dict[str, Any]:
    """查看智能体状态"""
    if not _agent:
        return {"status": "not_born", "message": "智能体尚未孵化"}
    
    status = _agent.get_status()
    
    # 获取最近的思考
    recent_thoughts = []
    if _agent.mind.thoughts:
        recent_thoughts = _agent.mind.thoughts[-3:]
    
    # 获取最近的经历
    recent_episodes = []
    if _agent.memory.episodes:
        recent_episodes = _agent.memory.episodes[-3:]
    
    return {
        "status": "alive" if status["alive"] else "dead",
        "cycles": status["cycles"],
        "body": status["body"],
        "memory": status["memory"],
        "emotion": status["current_emotion"],
        "recent_thoughts": recent_thoughts,
        "recent_episodes": recent_episodes,
    }


@router.post("/talk")
async def talk_to_agent(request: dict) -> dict[str, Any]:
    """与智能体对话 - 不是命令，而是交流"""
    global _agent
    
    if not _agent:
        return JSONResponse(
            {"status": "error", "message": "智能体尚未孵化，请先调用 /agent/spawn"},
            status_code=400
        )
    
    message = request.get("message", "")
    
    # 记录这次交互
    _agent.memory.add_episode(
        action=f"与用户对话: {message}",
        result="收到用户消息",
        emotion="engaged"
    )
    
    # 让智能体思考如何回应
    perception = f"用户对你说: '{message}'"
    thought = await _agent.mind.think(perception, _agent.memory, _agent.body)
    
    return {
        "status": "responded",
        "your_message": message,
        "agent_observation": thought.get("observation", ""),
        "agent_thought": thought.get("thought", ""),
        "agent_emotion": thought.get("emotion", ""),
        "agent_response": thought.get("next_action", ""),
    }


@router.post("/kill")
async def kill_agent() -> dict[str, Any]:
    """终止智能体 - 让它安息"""
    global _agent
    
    if not _agent:
        return {"status": "not_born", "message": "智能体不存在"}
    
    _agent.stop()
    cycles = _agent.cycle_count
    _agent = None
    
    return {
        "status": "dead",
        "message": "智能体已终止",
        "lifetime_cycles": cycles,
    }


@router.get("/memory")
async def agent_memory() -> dict[str, Any]:
    """查看智能体的记忆"""
    if not _agent:
        return {"status": "not_born"}
    
    return {
        "episodes": _agent.memory.episodes[-20:],  # 最近20条经历
        "facts": _agent.memory.facts,
        "skills": _agent.memory.skills,
    }


@router.get("/thoughts")
async def agent_thoughts() -> dict[str, Any]:
    """查看智能体的想法流"""
    if not _agent:
        return {"status": "not_born"}
    
    return {
        "thoughts": _agent.mind.thoughts[-20:],
        "reflections": _agent.mind.reflections[-10:],
    }


# WebSocket - 实时观察智能体的思维
@router.websocket("/ws")
async def agent_websocket(websocket: WebSocket):
    """WebSocket连接 - 实时观察智能体"""
    await websocket.accept()
    
    try:
        while True:
            if _agent and _agent.running:
                # 发送当前状态
                status = _agent.get_status()
                recent_thought = _agent.mind.thoughts[-1] if _agent.mind.thoughts else {}
                
                await websocket.send_json({
                    "type": "heartbeat",
                    "cycle": status["cycles"],
                    "emotion": status["current_emotion"],
                    "body": status["body"],
                    "thought": recent_thought,
                })
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass
