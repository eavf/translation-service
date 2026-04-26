import re
import logging
import threading
import time
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)

SUPPORTED_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("sk", "fr"), ("fr", "sk"),
    ("sk", "en"), ("en", "sk"),
    ("en", "fr"), ("fr", "en"),
    ("ar", "fr"), ("fr", "ar"),
})

_CHUNK_CACHE_MAX = 256


class Translator:
    """Single-direction model: one language pair, one MarianMT instance."""

    def __init__(self, model_id: str, chunk_size: int = 450) -> None:
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.model: Optional[MarianMTModel] = None
        self.tokenizer: Optional[MarianTokenizer] = None
        self.device = "cpu"
        self.is_loaded = False
        self._cache: dict[str, str] = {}

    def load(self) -> None:
        logger.info("Loading model: %s", self.model_id)
        t0 = time.monotonic()
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_id)
        self.model = MarianMTModel.from_pretrained(self.model_id)
        self.model.eval()
        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.2fs | device: %s", elapsed, self.device)
        self.is_loaded = True

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit within the model's token budget.

        Strategy (in order):
          1. Split on paragraph breaks (double newlines).
          2. If a paragraph still exceeds chunk_size, split on sentence boundaries.
          3. If a single sentence exceeds chunk_size, hard-split by character count.
        """
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= self.chunk_size:
                chunks.append(para)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""

            for sentence in sentences:
                candidate = (current + " " + sentence).strip() if current else sentence
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    while len(sentence) > self.chunk_size:
                        chunks.append(sentence[: self.chunk_size])
                        sentence = sentence[self.chunk_size :]
                    current = sentence

            if current:
                chunks.append(current)

        return chunks if chunks else [text.strip()]

    def _translate_batch(self, chunks: list[str]) -> list[str]:
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=2,
                early_stopping=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def translate(self, text: str) -> str:
        chunks = self.chunk_text(text)
        results: list[str] = [""] * len(chunks)

        uncached_indices = [i for i, c in enumerate(chunks) if c not in self._cache]

        for i, chunk in enumerate(chunks):
            if chunk in self._cache:
                results[i] = self._cache[chunk]

        if uncached_indices:
            uncached_chunks = [chunks[i] for i in uncached_indices]
            new_translations = self._translate_batch(uncached_chunks)
            for idx, chunk, translation in zip(uncached_indices, uncached_chunks, new_translations):
                results[idx] = translation
                if len(self._cache) >= _CHUNK_CACHE_MAX:
                    del self._cache[next(iter(self._cache))]
                self._cache[chunk] = translation

        return "\n\n".join(results)


class TranslatorRegistry:
    """Lazily loads one Translator per language pair on first use."""

    def __init__(self, chunk_size: int = 450) -> None:
        self.chunk_size = chunk_size
        self._translators: dict[tuple[str, str], Translator] = {}
        self._registry_lock = threading.Lock()
        self._pair_locks: dict[tuple[str, str], threading.Lock] = {}

    def _get_pair_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._registry_lock:
            if key not in self._pair_locks:
                self._pair_locks[key] = threading.Lock()
            return self._pair_locks[key]

    def get_translator(self, source_lang: str, target_lang: str) -> Translator:
        key = (source_lang, target_lang)
        if key in self._translators:
            return self._translators[key]
        lock = self._get_pair_lock(key)
        with lock:
            if key not in self._translators:
                model_id = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                t = Translator(model_id=model_id, chunk_size=self.chunk_size)
                t.load()
                self._translators[key] = t
        return self._translators[key]

    def translate(self, source_lang: str, target_lang: str, text: str) -> tuple[str, str]:
        """Returns (translated_text, model_id)."""
        translator = self.get_translator(source_lang, target_lang)
        return translator.translate(text), translator.model_id

    @property
    def loaded_models(self) -> list[dict]:
        return [
            {"pair": f"{src}-{tgt}", "model_id": t.model_id, "device": t.device}
            for (src, tgt), t in self._translators.items()
        ]
