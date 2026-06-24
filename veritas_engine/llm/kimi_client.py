from __future__ import annotations

import os
from typing import Any

import httpx

from veritas_engine.core.config import get_config


class KimiClient:
    """Kimi (Moonshot AI) / DeepSeek API client."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.deepseek.com/v1"):
        # 从环境变量或.env文件获取 DeepSeek API Key
        from dotenv import load_dotenv
        load_dotenv("/home/agent/kimi_truth_engine/project/.env")
        
        # 直接读取.env文件
        with open("/home/agent/kimi_truth_engine/project/.env", "r") as f:
            for line in f:
                if line.startswith("DEEPSEEK_API_KEY="):
                    env_api_key = line.strip().split("=", 1)[1]
                    break
            else:
                env_api_key = ""
        
        self.api_key = api_key or env_api_key
        self.base_url = base_url
        self.model = "deepseek-chat"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=60.0,
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Send chat completion request."""
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()
