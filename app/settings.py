from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_chars_per_request: int = 10000
    # Internal chunk size fed to the model tokenizer (characters, not tokens).
    # OPUS-MT models have a 512-token input limit; 450 chars is a safe ceiling
    # for Slovak/French/English where the char-to-token ratio is roughly 1:1.5.
    chunk_size: int = 450
    log_text: bool = False
    allowed_origins: str = "*"

    model_config = {"env_file": ".env"}


settings = Settings()
