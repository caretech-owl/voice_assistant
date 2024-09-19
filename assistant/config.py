from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from gerd.models.gen import GenerationConfig

PROJECT_DIR = Path(__file__).parent.parent


class STTConfig(BaseSettings):
    model: str
    pause_threshold: float
    non_speaking_duration: float
    timeout: int
    phrase_time_limit: int
    language: str


class TTSConfig(BaseSettings):
    provider: str
    model: str
    language: str
    speaker: str
    split_sentences: bool


class WakeWordConfig(BaseSettings):
    chunk_size: int
    model: str
    inference_framework: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    llm: GenerationConfig
    stt: STTConfig
    tts: TTSConfig
    wakeword: WakeWordConfig


with Path(PROJECT_DIR, "config", "config.json").open() as f:
    CONFIG = Settings.model_validate_json(f.read())
