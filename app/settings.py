from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # If set, this model is used as the primary backend (e.g. facebook/nllb-200-distilled-600M).
    # If empty, the service falls back to per-pair Helsinki-NLP OPUS-MT models.
    translation_model: str = ""
    # If TRANSLATION_MODEL fails to load, the service automatically falls back to
    # per-pair OPUS-MT auto-detection (Helsinki-NLP/opus-mt-{src}-{tgt}).
    # Set this to document the intended fallback in your .env / docker-compose.yml.
    fallback_translation_model: str = ""

    max_chars_per_request: int = 10000
    # Internal chunk size fed to the model tokenizer (characters, not tokens).
    # OPUS-MT: 512-token limit → 450 chars is safe.
    # NLLB: recommend ≤ 400 chars (~250 tokens) per the model card.
    chunk_size: int = 450
    log_text: bool = False
    allowed_origins: str = "*"

    model_config = {"env_file": ".env"}


settings = Settings()
