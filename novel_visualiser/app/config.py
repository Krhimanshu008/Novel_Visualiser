"""
Novel Visualiser — Application Configuration

Loads environment variables and provides typed settings for all subsystems.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """SQLite / PostgreSQL connection settings."""

    database_path: str = Field(
        default="./novel_visualiser.db",
        description="Path to the local SQLite database file.",
    )
    echo_sql: bool = Field(
        default=False,
        description="Echo all SQL statements to stdout for debugging.",
    )

    model_config = {"env_prefix": "DB_", "env_file": ".env", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM provider configuration (Gemini API or local Ollama)."""

    # --- Google Gemini (primary & secondary) ---
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_api_key_secondary: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY_SECONDARY")
    gemini_model: str = Field(default="gemini-2.5-flash")

    # --- Groq (Ultra-Fast LPUs) ---
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="qwen/qwen3.8-27b")

    # --- OpenRouter (Multi-Model Agentic Gateway) ---
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="nvidia/nemotron-3.5-lightning:free")

    # --- Kaggle Agentic AI & Model Proxy ---
    kaggle_api_token: Optional[str] = Field(default=None, alias="KAGGLE_API_TOKEN")
    kaggle_key: Optional[str] = Field(default=None, alias="KAGGLE_KEY")
    kaggle_username: Optional[str] = Field(default=None, alias="KAGGLE_USERNAME")

    # --- Local Ollama (fallback / free tier) ---
    ollama_base_url: Optional[str] = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:14b")

    # --- Shared ---
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192)

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore", "populate_by_name": True}


class ComfyUISettings(BaseSettings):
    """ComfyUI server connection settings (supports local, Colab, or Kaggle tunnel URLs)."""

    url: Optional[str] = Field(
        default=None,
        description="Full URL if using Kaggle/Colab tunnel (e.g. https://xxxx.trycloudflare.com or https://xxxx.ngrok-free.app)",
    )
    gpu0_url: Optional[str] = Field(
        default=None,
        description="Permanent URL for GPU 0 worker (e.g. https://gpu0.krhimanshu.in)",
    )
    gpu1_url: Optional[str] = Field(
        default=None,
        description="Permanent URL for GPU 1 worker (e.g. https://gpu1.krhimanshu.in)",
    )
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8188)
    use_ssl: bool = Field(default=False)
    client_id: str = Field(default="novel_visualiser")
    timeout_seconds: int = Field(default=600)
    allow_mock_fallback: bool = Field(
        default=False,
        description="If True, permits local mock canvas generation when ComfyUI is offline (test mode only).",
    )

    @property
    def base_url(self) -> str:
        if self.gpu0_url:
            return self.gpu0_url.rstrip("/")
        if self.url:
            return self.url.rstrip("/")
        scheme = "https" if self.use_ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @property
    def cluster_urls(self) -> list[str]:
        """Return list of active cluster URLs for Dual-GPU dispatching."""
        urls: list[str] = []
        if self.gpu0_url:
            urls.append(self.gpu0_url.rstrip("/"))
        elif self.url:
            urls.append(self.url.rstrip("/"))
        if self.gpu1_url:
            urls.append(self.gpu1_url.rstrip("/"))
        if not urls:
            urls.append(self.base_url)
        return urls

    @property
    def ws_url(self) -> str:
        from urllib.parse import urlparse
        if self.url:
            parsed = urlparse(self.url)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            return f"{ws_scheme}://{parsed.netloc}/ws?clientId={self.client_id}"
        scheme = "wss" if self.use_ssl else "ws"
        return f"{scheme}://{self.host}:{self.port}/ws?clientId={self.client_id}"

    model_config = {"env_prefix": "COMFYUI_", "env_file": ".env", "extra": "ignore"}


class TTSSettings(BaseSettings):
    """Text-to-Speech engine configuration."""

    engine: str = Field(
        default="local_xtts",
        description="TTS engine: 'local_xtts', 'elevenlabs', 'sarvam'",
    )
    voice_dir: str = Field(default="./assets/voices")
    elevenlabs_api_key: Optional[str] = Field(default=None)
    elevenlabs_voice_id: Optional[str] = Field(default=None)

    model_config = {"env_prefix": "TTS_", "env_file": ".env", "extra": "ignore"}


class ProjectSettings(BaseSettings):
    """Top-level project paths and settings."""

    project_name: str = Field(default="novel_visualiser")

    # --- Paths ---
    base_dir: str = Field(default=".")
    media_root: str = Field(default="./output")
    templates_dir: str = Field(default="./templates/comfyui")
    novels_dir: str = Field(default="./data/novels")
    references_dir: str = Field(default="./data/references")

    # --- Google Drive Sync ---
    gdrive_sync_enabled: bool = Field(default=False)
    gdrive_remote_path: str = Field(default="Novel_Visualiser_Backup")

    model_config = {"env_prefix": "PROJECT_", "env_file": ".env", "extra": "ignore"}

    def ensure_directories(self) -> None:
        """Create all required output directories if they don't exist."""
        dirs = [
            Path(self.media_root) / "generated_images",
            Path(self.media_root) / "depth_maps",
            Path(self.media_root) / "audio",
            Path(self.media_root) / "renders",
            Path(self.references_dir),
            Path(self.novels_dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Master settings container aggregating all subsystem configs."""

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    comfyui: ComfyUISettings = Field(default_factory=ComfyUISettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    project: ProjectSettings = Field(default_factory=ProjectSettings)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# --- Singleton ---
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

