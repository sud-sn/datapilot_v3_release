"""
DataPilot – Model Manager
Handles sequential model loading via Ollama's keep_alive parameter.

  Qwen 2.5 Coder  →  SQL generation  →  keep_alive = -1  (always hot)
  Llama 3.1       →  MD generation   →  keep_alive =  0  (load, use, unload)

Ollama automatically manages VRAM: a model with keep_alive=0 is evicted
immediately after the request completes, freeing memory for the next model.
A model with keep_alive=-1 stays resident until the server is stopped.
"""
from __future__ import annotations
import logging
from typing import Optional

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Thin wrapper around the Ollama /api/generate endpoint with explicit
    keep_alive control for the two-model sequential loading pattern.

    Usage:
        mgr = ModelManager()
        await mgr.start()

        # Markdown generation (Llama — loads then unloads)
        md = await mgr.generate_markdown(prompt, system_prompt)

        # SQL generation is handled by OllamaProvider; this class just
        # pre-warms Qwen so it is resident before the first query arrives.
        await mgr.warmup_sql_model()

        await mgr.stop()
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._settings.llm_timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    # ── Markdown / interview model (Llama) ────────────────────────────────────

    async def generate_markdown(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> str:
        """
        Call Llama for markdown generation.
        keep_alive=0 means the model is evicted from VRAM immediately after
        this request completes, freeing memory for the SQL model.
        """
        model = self._settings.llm_model_interview
        logger.info("Loading interview model: %s", model)

        payload = {
            "model":      model,
            "prompt":     user_prompt,
            "system":     system_prompt,
            "stream":     False,
            "keep_alive": 0,          # ← unload immediately after use
            "options": {
                "temperature": 0.1,   # Low but not zero — creativity helps here
                "num_predict": 4096,  # Markdown can be long
            },
        }

        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            logger.info("Interview model unloaded after generation.")
            return result
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Interview model ({model}) returned HTTP {exc.response.status_code}. "
                "Is the model pulled? Run: ollama pull " + model
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._settings.ollama_base_url}. "
                "Is Ollama running?"
            ) from exc

    # ── SQL model warm-up (Qwen) ──────────────────────────────────────────────

    async def warmup_sql_model(self) -> None:
        """
        Pre-load the SQL model into VRAM at startup so the first query
        doesn't suffer a cold-start delay.
        keep_alive=-1 keeps it resident for the lifetime of the server.
        """
        model = self._settings.llm_model_primary
        logger.info("Warming up SQL model: %s", model)
        payload = {
            "model":      model,
            "prompt":     "SELECT 1;",
            "system":     "You are a SQL expert.",
            "stream":     False,
            "keep_alive": -1,         # ← stay resident forever
            "options": {"num_predict": 10},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            logger.info("SQL model warm: %s", model)
        except Exception as exc:
            logger.warning("SQL model warm-up failed (%s): %s", model, exc)

    # ── Model availability check ──────────────────────────────────────────────

    async def check_models(self) -> dict[str, bool]:
        """Return availability status for both models."""
        results: dict[str, bool] = {}
        models_to_check = [
            self._settings.llm_model_interview,
            self._settings.llm_model_primary,
            self._settings.llm_model_fallback,
        ]
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            available = {m["name"] for m in resp.json().get("models", [])}
            for model in models_to_check:
                results[model] = any(a.startswith(model.split(":")[0]) for a in available)
        except Exception as exc:
            logger.warning("Could not check Ollama models: %s", exc)
            for model in models_to_check:
                results[model] = False
        return results
