"""LLM client supporting multiple providers: Ollama, OpenAI, and custom API-compatible endpoints."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from veritas_engine.core.config import get_config
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import LLMError, LLMTimeoutError, ToolCallError

logger = get_logger("veritas.llm")


class BaseLLMClient:
    """LLM 客户端基类——定义统一接口."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def generate(self, prompt: str, model: str | None = None, **options: Any) -> str:
        raise NotImplementedError

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class OllamaClient(BaseLLMClient):
    """Ollama LLM 客户端."""

    def __init__(self) -> None:
        self.config = get_config().llm
        self.base_url = self.config.base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        model_name = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise LLMTimeoutError(model_name, self.config.timeout)
        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise LLMError(str(e))

    async def generate(self, prompt: str, model: str | None = None, **options: Any) -> str:
        model_name = model or self.config.model
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.TimeoutException:
            raise LLMTimeoutError(model_name, self.config.timeout)
        except Exception as e:
            raise LLMError(str(e))

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        embed_model = model or self.config.embedding_model
        results = []
        for text in texts:
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": embed_model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                results.append(data.get("embedding", []))
            except Exception as e:
                logger.error("Embedding error: %s", e, extra={"layer": "llm"})
                results.append([0.0] * 768)
        return results

    async def close(self) -> None:
        await self.client.aclose()


class OpenAIClient(BaseLLMClient):
    """OpenAI API 兼容客户端——支持 OpenAI、Azure、以及任何兼容 OpenAI API 的端点."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.config = get_config().llm
        self.api_key = api_key or self._get_api_key()
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4"
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _get_api_key(self) -> str:
        """从环境变量获取 API key."""
        import os
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.warning("OPENAI_API_KEY not set, using dummy key", extra={"layer": "llm"})
            return "dummy-key"
        return key

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        model_name = model or self.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # 统一返回格式（兼容 Ollama 格式）
            return {
                "message": {
                    "role": data["choices"][0]["message"]["role"],
                    "content": data["choices"][0]["message"].get("content", ""),
                    "tool_calls": data["choices"][0]["message"].get("tool_calls", []),
                },
                "done": True,
                "model": model_name,
            }
        except httpx.TimeoutException:
            raise LLMTimeoutError(model_name, self.config.timeout)
        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise LLMError(str(e))

    async def generate(self, prompt: str, model: str | None = None, **options: Any) -> str:
        """使用 chat completions 模拟 generate."""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat(messages, model=model, **options)
        return result.get("message", {}).get("content", "")

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        embed_model = model or "text-embedding-3-small"
        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                json={"model": embed_model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error("OpenAI embedding error: %s", e, extra={"layer": "llm"})
            return [[0.0] * 1536 for _ in texts]

    async def close(self) -> None:
        await self.client.aclose()


class LLMClient:
    """统一 LLM 客户端——根据配置自动选择后端."""

    PROVIDERS = {
        "ollama": OllamaClient,
        "openai": OpenAIClient,
    }

    def __init__(self, provider: str | None = None) -> None:
        self.config = get_config().llm
        self.provider = provider or self.config.provider
        self._client: BaseLLMClient | None = None
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "ollama":
            self._client = OllamaClient()
        elif self.provider == "openai":
            self._client = OpenAIClient()
        else:
            # 尝试作为自定义 OpenAI 兼容端点
            logger.warning("Unknown provider '%s', falling back to OpenAI-compatible", self.provider)
            self._client = OpenAIClient(
                base_url=self.config.base_url,
                model=self.config.model,
            )

    async def chat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self._client:
            raise LLMError("LLM client not initialized")
        return await self._client.chat(*args, **kwargs)

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        if not self._client:
            raise LLMError("LLM client not initialized")
        return await self._client.generate(*args, **kwargs)

    async def embeddings(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        if not self._client:
            raise LLMError("LLM client not initialized")
        return await self._client.embeddings(*args, **kwargs)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    def switch_provider(self, provider: str, **kwargs: Any) -> None:
        """切换 LLM 提供商."""
        if self._client:
            asyncio.create_task(self._client.close())
        self.provider = provider
        if provider == "ollama":
            self._client = OllamaClient()
        elif provider == "openai":
            self._client = OpenAIClient(**kwargs)
        else:
            self._client = OpenAIClient(**kwargs)
        logger.info("Switched LLM provider to %s", provider, extra={"layer": "llm"})


class ToolRegistry:
    """工具注册表——管理 LLM 可调用的工具."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """注册工具."""
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._handlers[name] = handler
        logger.info("Tool registered: %s", name, extra={"layer": "llm"})

    def get_tools(self) -> list[dict[str, Any]]:
        """获取所有工具定义（用于传给 LLM）."""
        return list(self._tools.values())

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用工具."""
        if name not in self._handlers:
            raise ToolCallError(name, f"Tool not registered")

        handler = self._handlers[name]
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(**arguments)
            else:
                return handler(**arguments)
        except Exception as e:
            raise ToolCallError(name, str(e))

    def list_tools(self) -> list[str]:
        """列出所有已注册工具名称."""
        return list(self._tools.keys())


import asyncio
