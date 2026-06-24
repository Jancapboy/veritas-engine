"""FastAPI server for Veritas Engine.

Provides REST API and WebSocket for real-time system status.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from veritas_engine.api.agent_routes import router as agent_router
from veritas_engine.core.engine import Engine
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.api")

# Global engine instance
_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理——启动/停止引擎."""
    global _engine
    _engine = Engine()
    await _engine.start()
    logger.info("API server started with engine", extra={"layer": "system"})
    yield
    await _engine.stop()
    _engine = None
    logger.info("API server stopped", extra={"layer": "system"})


app = FastAPI(
    title="Veritas Engine API",
    description="Autonomous Evolutionary Intelligence Agent Framework",
    version="0.1.0-alpha",
    lifespan=lifespan,
)

app.include_router(agent_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST API ──

@app.get("/health")
async def health() -> dict[str, Any]:
    """健康检查."""
    return {"status": "healthy", "engine": "running" if _engine and _engine._running else "stopped"}


@app.post("/think")
async def think(request: Request) -> dict[str, Any]:
    """多模型思维推理入口
    
    Request body: {"problem": "问题描述", "context": {}}
    """
    data = await request.json()
    problem = data.get("problem", "")
    context = data.get("context", {})
    
    if not problem:
        return JSONResponse({"error": "Missing 'problem' field"}, status_code=400)
    
    try:
        from veritas_engine.models import ModelThinker
        thinker = ModelThinker()
        result = await thinker.think(problem, context)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/llm/status")
async def llm_status() -> dict[str, Any]:
    """获取 LLM 层状态."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    return {
        "provider": _engine.config.llm.provider if _engine.config else "ollama",
        "model": _engine.config.llm.model if _engine.config else "llama3.1",
        "base_url": _engine.config.llm.base_url if _engine.config else "http://localhost:11434",
        "available": True,
    }


@app.post("/llm/chat")
async def llm_chat(request: dict[str, Any]) -> dict[str, Any]:
    """直接调用 LLM chat."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    messages = request.get("messages", [])
    model = request.get("model")
    temperature = request.get("temperature")
    try:
        result = await _engine.llm.chat(messages, model=model, temperature=temperature)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/llm/switch")
async def llm_switch(request: dict[str, Any]) -> dict[str, Any]:
    """切换 LLM 提供商."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    provider = request.get("provider", "ollama")
    api_key = request.get("api_key", "")
    model = request.get("model")

    if not api_key and provider != "ollama":
        return JSONResponse({"error": "API key required"}, status_code=400)

    if _engine.config:
        _engine.config.set_llm_provider(provider, api_key, model)

    # 重新初始化 LLM 客户端
    from veritas_engine.llm import KimiClient, OpenAIClient, OllamaClient
    if provider == "kimi":
        _engine.llm = KimiClient(api_key=api_key, model=model or "moonshot-v1-8k")
    elif provider in ("openai", "deepseek"):
        _engine.llm = OpenAIClient(
            api_key=api_key,
            model=model or ("gpt-4o-mini" if provider == "openai" else "deepseek-chat"),
            base_url=_engine.config.llm.base_url,
        )
    else:
        _engine.llm = OllamaClient()

    return {"success": True, "provider": provider, "model": model or "default"}


@app.get("/status")
async def status() -> dict[str, Any]:
    """获取系统状态."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    system_status = await _engine.get_status()
    return system_status.model_dump()


@app.get("/layers")
async def layers() -> dict[str, Any]:
    """获取六层架构状态."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    status = await _engine.get_status()
    return {
        "layers": status.layer_status,
        "emotional_state": status.emotional_state.model_dump() if status.emotional_state else None,
    }


@app.get("/emotions")
async def emotions() -> dict[str, Any]:
    """获取当前情感状态."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    state = _engine.daemon.get_state()
    return {
        "curiosity": state.curiosity,
        "urgency": state.urgency,
        "frustration": state.frustration,
        "achievement": state.achievement,
        "value_weights": state.value_weights,
        "epsilon": state.epsilon,
        "exploration_bias": state.exploration_bias,
        "exploitation_bias": state.exploitation_bias,
        "value_function": state.value_function_str,
    }


@app.post("/goals")
async def run_goal(goal: dict[str, Any]) -> dict[str, Any]:
    """执行目标推理.

    Request body: {"goal": "优化MES排产效率"}
    """
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    goal_text = goal.get("goal", "")
    if not goal_text:
        return JSONResponse({"error": "Missing 'goal' field"}, status_code=400)
    result = await _engine.run_goal(goal_text)
    return result


@app.get("/strategies")
async def list_strategies() -> dict[str, Any]:
    """列出所有策略."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    return {
        "total": len(_engine._strategies),
        "strategies": [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                "confidence": s.confidence,
                "source": s.source.value if hasattr(s.source, "value") else str(s.source),
            }
            for s in _engine._strategies
        ],
    }


@app.get("/memories")
async def list_memories(query: str = "", top_k: int = 10) -> dict[str, Any]:
    """搜索向量记忆."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    results = _engine.noosphere.vector_search(query, top_k)
    return {"query": query, "results": results, "total": len(results)}


@app.get("/working-memory")
async def working_memory() -> dict[str, Any]:
    """获取工作记忆内容."""
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    wm = _engine.noosphere.get_working_memory()
    return wm.summary()


@app.post("/perception")
async def emit_perception(event: dict[str, Any]) -> dict[str, Any]:
    """手动发送感知事件.

    Request body: {"source": "api", "event_type": "human_command", "payload": {...}}
    """
    if not _engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=503)
    from veritas_engine.core.models import PerceptionEvent, EventType
    event_type = event.get("event_type", "human_command")
    try:
        et = EventType(event_type)
    except ValueError:
        et = EventType.HUMAN_COMMAND
    pe = PerceptionEvent(
        source=event.get("source", "api"),
        event_type=et,
        payload=event.get("payload", {}),
        metadata=event.get("metadata", {}),
    )
    await _engine.sensorium.emit(pe)
    return {"success": True, "event_id": pe.id}


# ── SPA Static Files ──

_frontend_path = "/home/agent/kimi_truth_engine/app/dist"

@app.get("/")
async def root():
    return FileResponse(os.path.join(_frontend_path, "index.html"))

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """SPA fallback: serve index.html for all non-API routes."""
    # API 路径不走 fallback
    api_prefixes = ("health", "status", "layers", "emotions", "goals", "strategies", "memories", "working-memory", "perception", "llm", "api", "ws")
    if full_path.startswith(api_prefixes):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    file_path = os.path.join(_frontend_path, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(_frontend_path, "index.html"))


# ── WebSocket ──

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket 实时状态推送.

    连接后每 2 秒推送一次系统状态。
    """
    await websocket.accept()
    logger.info("WebSocket client connected", extra={"layer": "system"})
    try:
        while True:
            if _engine and _engine._running:
                status = await _engine.get_status()
                await websocket.send_json({
                    "type": "status",
                    "timestamp": status.timestamp.isoformat(),
                    "layer_status": status.layer_status,
                    "emotional_state": status.emotional_state.model_dump() if status.emotional_state else None,
                    "knowledge_count": status.knowledge_count,
                    "pending_strategies": status.pending_strategies,
                    "executed_strategies": status.executed_strategies,
                })
            else:
                await websocket.send_json({"type": "error", "message": "Engine not running"})
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", extra={"layer": "system"})
    except Exception as e:
        logger.error("WebSocket error: %s", e, extra={"layer": "system"})
        await websocket.close()
