"""
Novel Visualiser — Kaggle Model Proxy Unified Client

Provides an interface to Kaggle's Model Proxy using the official
`kaggle_benchmarks` engine to run frontier models (gemini-3-flash-preview,
claude-sonnet-5, gpt-5.4, deepseek-r1, qwen3) leveraging the user's
$10/day spend quota.

Features:
- Official Kaggle Model Proxy SDK (`kaggle_benchmarks`)
- Structured Pydantic extraction out-of-the-box
- Multimodal / image inspection for Visual Consistency & Gate 3 QC
- Automatic credential reload and expiry guards
- Fallback to direct Gemini API
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

# Ensure Windows stdout handles emoji logging from kaggle_benchmarks without charmap errors
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel

from kaggle_ai_engine.config import get_kaggle_settings

logger = logging.getLogger("kaggle_ai_engine.client")
T = TypeVar("T", bound=BaseModel)

# Try importing official kaggle_benchmarks engine (available in Kaggle environment)
kbench: Any = None
kbench_images: Any = None
KBENCH_AVAILABLE = False

try:
    import importlib.util
    if importlib.util.find_spec("kaggle_benchmarks") is not None:
        import kaggle_benchmarks as kbench  # type: ignore[import-not-found, import-untyped]
        from kaggle_benchmarks.content_types import images as kbench_images  # type: ignore[import-not-found, import-untyped]
        KBENCH_AVAILABLE = True
except Exception as e:
    logger.debug("kaggle_benchmarks SDK not imported directly: %s", e)
    kbench = None
    kbench_images = None
    KBENCH_AVAILABLE = False


class KaggleModelProxyClient:
    """
    Client for interacting with models via Kaggle's Model Proxy.
    Draws against the user's daily $10 allowance.
    """

    def __init__(self, timeout: int = 120):
        self.settings = get_kaggle_settings()
        self.timeout = timeout
        self._ensure_env_synced()

    def _ensure_env_synced(self) -> None:
        """Sync critical Kaggle Benchmarks environment variables."""
        if not os.getenv("LLM_DEFAULT"):
            os.environ["LLM_DEFAULT"] = self.settings.llm_default
        if not os.getenv("LLM_DEFAULT_EVAL"):
            os.environ["LLM_DEFAULT_EVAL"] = self.settings.llm_default_eval

    def _get_kbench_llm(self, model: Optional[str] = None):
        """Retrieve appropriate kbench LLM object."""
        if not KBENCH_AVAILABLE or kbench is None:
            return None
        self._ensure_env_synced()
        try:
            if model and hasattr(kbench, "llms") and model in kbench.llms:
                return kbench.llms[model]
            return kbench.llm
        except Exception as e:
            logger.debug("Could not resolve kbench LLM: %s", e)
            return None

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Execute chat completion via Kaggle Model Proxy.
        """
        llm = self._get_kbench_llm(model)
        if llm is not None:
            try:
                # Format conversation history
                full_prompt = "\n".join(f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in messages)
                resp = llm.prompt(full_prompt)
                return str(resp)
            except Exception as err:
                logger.warning("Kaggle SDK prompt error: %s; falling back", err)

        # Fallback 1: Ultra-fast Groq LPU inference if configured
        if getattr(self.settings, "groq_api_key", None):
            try:
                return self._groq_direct_fallback(messages, model=model, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                logger.warning("Groq direct fallback failed: %s; trying Gemini/OpenRouter", e)

        # Fallback 2: Direct Gemini API if configured
        if self.settings.google_api_key or getattr(self.settings, "google_api_key_secondary", None):
            try:
                return self._gemini_direct_fallback(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                logger.warning("Gemini direct fallback failed: %s; trying OpenRouter", e)

        # Fallback 3: OpenRouter multi-model gateway
        if getattr(self.settings, "openrouter_api_key", None):
            return self._openrouter_direct_fallback(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        raise RuntimeError("Could not execute chat completion through Kaggle Model Proxy or fallback.")

    def structured_extract(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
    ) -> T:
        """
        Extract strongly-typed Pydantic model directly via Kaggle Model Proxy.
        """
        llm = self._get_kbench_llm(model)
        if llm is not None:
            try:
                sys_prefix = f"{system_instruction}\n\n" if system_instruction else ""
                full_query = f"{sys_prefix}{prompt}"
                extracted = llm.prompt(full_query, schema=schema)
                if isinstance(extracted, schema):
                    return extracted
                if isinstance(extracted, dict):
                    return schema.model_validate(extracted)
                if isinstance(extracted, str):
                    clean = extracted.strip("` \n\r")
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                    return schema.model_validate_json(clean)
            except Exception as err:
                logger.warning("Kaggle SDK structured prompt error: %s; attempting raw JSON fallback", err)

        # Fallback parsing
        json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
        sys_msg = (
            (system_instruction or "You are an expert AI extraction agent.")
            + f"\nYou MUST respond ONLY with valid JSON matching this schema:\n{json_schema_str}\n"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        raw = self.chat_completion(messages, model=model, temperature=0.1, response_format="json")
        clean_text = raw.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        return schema.model_validate_json(clean_text.strip())

    def multimodal_inspect(
        self,
        prompt: str,
        image_paths: list[Path | str],
        model: Optional[str] = None,
        schema: Optional[Type[T]] = None,
    ) -> Any:
        """
        Multimodal VLM inspection of images using Kaggle Model Proxy.
        Used for Gate 3 QC (verifying actual generated pixels).
        """
        llm = self._get_kbench_llm(model)
        if llm is not None and kbench_images is not None and image_paths:
            try:
                valid_images = [Path(p) for p in image_paths if Path(p).exists()]
                if valid_images:
                    primary_img = kbench_images.from_path(str(valid_images[0]))
                    if schema:
                        return llm.prompt(prompt, image=primary_img, schema=schema)
                    return str(llm.prompt(prompt, image=primary_img))
            except Exception as exc:
                logger.warning("kbench multimodal error: %s; falling back", exc)

        # Fallback to direct Gemini Vision
        return self._gemini_multimodal_fallback(prompt, image_paths)

    def _gemini_direct_fallback(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        """Direct Google Gemini API fallback with primary & secondary keys."""
        import requests
        keys = [self.settings.google_api_key]
        if getattr(self.settings, "google_api_key_secondary", None):
            keys.append(self.settings.google_api_key_secondary)
        candidate_models = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-pro-latest"]

        contents = []
        for m in messages:
            role = "user" if m["role"] in ["user", "system"] else "model"
            contents.append({"role": role, "parts": [{"text": str(m["content"])}]})

        body = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        last_error = None
        for key in keys:
            if not key:
                continue
            for mod in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key}"
                try:
                    resp = requests.post(url, json=body, timeout=min(5, self.timeout))
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    last_error = f"{resp.status_code}: {resp.text}"
                except Exception as e:
                    last_error = str(e)

        raise RuntimeError(f"Gemini fallback failed across models: {last_error}")

    def _gemini_multimodal_fallback(self, prompt: str, image_paths: list[Path | str]) -> str:
        """Direct Gemini multimodal API call with multi-key failover."""
        import requests
        keys = [self.settings.google_api_key]
        if getattr(self.settings, "google_api_key_secondary", None):
            keys.append(self.settings.google_api_key_secondary)

        candidate_models = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-pro-latest"]

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for p in image_paths:
            path = Path(p)
            if path.exists():
                ext = path.suffix.lower().lstrip(".")
                mime = f"image/{ext}" if ext in ["png", "jpeg", "webp"] else "image/jpeg"
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
                    }
                })

        body = {"contents": [{"parts": parts}]}
        last_err = None
        for key in keys:
            if not key:
                continue
            for mod in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key}"
                try:
                    resp = requests.post(url, json=body, timeout=self.timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    last_err = f"{resp.status_code}: {resp.text}"
                except Exception as e:
                    last_err = str(e)
        raise RuntimeError(f"Gemini multimodal fallback failed: {last_err}")


    def _openrouter_direct_fallback(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Direct OpenRouter API call for chat completion."""
        import requests
        api_key = self.settings.openrouter_api_key
        target_model = model or getattr(self.settings, "openrouter_model", "nvidia/nemotron-3.5-lightning:free")
        if target_model.startswith("kbench/") or "/" not in target_model:
            target_model = getattr(self.settings, "openrouter_model", "nvidia/nemotron-3.5-lightning:free")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/Krhimanshu008/Novel_Visualiser",
            "X-Title": "Novel Visualiser",
        }
        body = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


    def _groq_direct_fallback(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Direct Groq LPU API call with multi-model failover."""
        import requests
        api_key = self.settings.groq_api_key
        primary_model = model or getattr(self.settings, "groq_model", "openai/gpt-oss-120b")
        groq_candidates = [
            primary_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        candidate_models = list(dict.fromkeys(groq_candidates))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for cand in candidate_models:
            body = {
                "model": cand,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=min(10, self.timeout),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                last_error = f"{cand} failed with HTTP {resp.status_code}: {resp.text}"
                logger.warning("Groq model %s error: %s; trying next Groq candidate", cand, resp.status_code)
            except Exception as e:
                last_error = str(e)
                logger.warning("Groq model %s exception: %s; trying next Groq candidate", cand, e)

        raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

    def transcribe_audio(self, audio_file: Path | str, model: str = "whisper-large-v3-turbo") -> str:
        """Transcribe speech audio via Groq Whisper API (whisper-large-v3-turbo)."""
        import requests
        api_key = self.settings.groq_api_key
        if not api_key:
            raise RuntimeError("GROQ_API_KEY required for Groq Whisper transcription.")

        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        headers = {"Authorization": f"Bearer {api_key}"}
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/mpeg")}
            data = {"model": model, "response_format": "text"}
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.text.strip()
