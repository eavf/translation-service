import re
import logging
import threading
import time
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)

# ── Language definitions ───────────────────────────────────────────────────────

# FLORES-200 codes used by NLLB models.
NLLB_LANG_MAP: dict[str, str] = {
    "sk": "slk_Latn",
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "ar": "arb_Arab",
}

# Pairs supported by Helsinki-NLP OPUS-MT (one model per direction).
OPUS_MT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("sk", "fr"), ("fr", "sk"),
    ("sk", "en"), ("en", "sk"),
    ("en", "fr"), ("fr", "en"),
    ("ar", "fr"), ("fr", "ar"),
})

# All combinations within NLLB_LANG_MAP — used when NLLB is the active backend.
NLLB_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (src, tgt)
    for src in NLLB_LANG_MAP
    for tgt in NLLB_LANG_MAP
    if src != tgt
)

_CACHE_MAX = 256


# ── Shared text chunker ────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int) -> list[str]:
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

        if len(para) <= chunk_size:
            chunks.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                while len(sentence) > chunk_size:
                    chunks.append(sentence[:chunk_size])
                    sentence = sentence[chunk_size:]
                current = sentence

        if current:
            chunks.append(current)

    return chunks if chunks else [text.strip()]


# ── OPUS-MT (Helsinki-NLP) — one model per language pair ──────────────────────

class OpusMTTranslator:
    def __init__(self, model_id: str, chunk_size: int = 450) -> None:
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.model: Optional[MarianMTModel] = None
        self.tokenizer: Optional[MarianTokenizer] = None
        self.device = "cpu"
        self.is_loaded = False
        self._cache: dict[str, str] = {}

    def load(self) -> None:
        logger.info("Loading OPUS-MT model: %s", self.model_id)
        t0 = time.monotonic()
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_id)
        self.model = MarianMTModel.from_pretrained(self.model_id)
        self.model.eval()
        logger.info("Model loaded in %.2fs | device: %s", time.monotonic() - t0, self.device)
        self.is_loaded = True

    def _translate_batch(self, chunks: list[str]) -> list[str]:
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer(
            chunks, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_length=512, num_beams=2, early_stopping=True)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def translate(self, text: str) -> str:
        chunks = chunk_text(text, self.chunk_size)
        results: list[str] = [""] * len(chunks)

        uncached_indices = [i for i, c in enumerate(chunks) if c not in self._cache]
        for i, c in enumerate(chunks):
            if c in self._cache:
                results[i] = self._cache[c]

        if uncached_indices:
            new = self._translate_batch([chunks[i] for i in uncached_indices])
            for idx, c, t in zip(uncached_indices, [chunks[i] for i in uncached_indices], new):
                results[idx] = t
                if len(self._cache) >= _CACHE_MAX:
                    del self._cache[next(iter(self._cache))]
                self._cache[c] = t

        return "\n\n".join(results)


# ── NLLB (Facebook/Meta) — one model for all language pairs ───────────────────

class NLLBTranslator:
    def __init__(self, model_id: str, chunk_size: int = 400) -> None:
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.is_loaded = False
        # Cache key includes direction because same text → different output per target lang.
        self._cache: dict[tuple[str, str, str], str] = {}

    def load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info("Loading NLLB model: %s", self.model_id)
        t0 = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.model.eval()
        logger.info("NLLB model loaded in %.2fs | device: %s", time.monotonic() - t0, self.device)
        self.is_loaded = True

    def _translate_batch(self, chunks: list[str], src_code: str, tgt_code: str) -> list[str]:
        assert self.tokenizer is not None and self.model is not None
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            chunks, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
                num_beams=2,
                early_stopping=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        src_code = NLLB_LANG_MAP[source_lang]
        tgt_code = NLLB_LANG_MAP[target_lang]
        chunks = chunk_text(text, self.chunk_size)
        results: list[str] = [""] * len(chunks)

        uncached_indices = [
            i for i, c in enumerate(chunks) if (c, source_lang, target_lang) not in self._cache
        ]
        for i, c in enumerate(chunks):
            if (c, source_lang, target_lang) in self._cache:
                results[i] = self._cache[(c, source_lang, target_lang)]

        if uncached_indices:
            new = self._translate_batch([chunks[i] for i in uncached_indices], src_code, tgt_code)
            for idx, c, t in zip(uncached_indices, [chunks[i] for i in uncached_indices], new):
                results[idx] = t
                if len(self._cache) >= _CACHE_MAX:
                    del self._cache[next(iter(self._cache))]
                self._cache[(c, source_lang, target_lang)] = t

        return "\n\n".join(results)


# ── Registry — routes requests to the right backend ───────────────────────────

class TranslatorRegistry:
    def __init__(
        self,
        chunk_size: int = 450,
        translation_model: str = "",
        fallback_model: str = "",
    ) -> None:
        self.chunk_size = chunk_size

        # NLLB primary backend (optional)
        self._nllb: Optional[NLLBTranslator] = None
        self._nllb_lock = threading.Lock()
        self._nllb_failed = False
        if translation_model:
            nllb_chunk = min(chunk_size, 400)
            self._nllb = NLLBTranslator(model_id=translation_model, chunk_size=nllb_chunk)
            logger.info("Primary backend: NLLB (%s, chunk_size=%d)", translation_model, nllb_chunk)
        else:
            logger.info("Primary backend: OPUS-MT per-pair")

        # OPUS-MT fallback (per-pair, lazy)
        self._fallback_model = fallback_model
        self._opus: dict[tuple[str, str], OpusMTTranslator] = {}
        self._registry_lock = threading.Lock()
        self._pair_locks: dict[tuple[str, str], threading.Lock] = {}

    @property
    def supported_pairs(self) -> frozenset[tuple[str, str]]:
        if self._nllb is not None and not self._nllb_failed:
            return NLLB_PAIRS
        return OPUS_MT_PAIRS

    @property
    def backend(self) -> str:
        if self._nllb is not None and not self._nllb_failed:
            return "nllb"
        return "opus-mt"

    def _load_nllb(self) -> NLLBTranslator:
        assert self._nllb is not None
        if self._nllb.is_loaded:
            return self._nllb
        with self._nllb_lock:
            if not self._nllb.is_loaded:
                self._nllb.load()
        return self._nllb

    def _get_pair_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._registry_lock:
            if key not in self._pair_locks:
                self._pair_locks[key] = threading.Lock()
            return self._pair_locks[key]

    def _get_opus(self, source_lang: str, target_lang: str) -> OpusMTTranslator:
        key = (source_lang, target_lang)
        if key in self._opus:
            return self._opus[key]
        model_id = self._fallback_model or f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        lock = self._get_pair_lock(key)
        with lock:
            if key not in self._opus:
                t = OpusMTTranslator(model_id=model_id, chunk_size=self.chunk_size)
                t.load()
                self._opus[key] = t
        return self._opus[key]

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        text: str,
        backend: Optional[str] = None,
    ) -> tuple[str, str]:
        """Returns (translated_text, model_id).

        backend: "nllb" | "opus-mt" | None (use configured default)
        """
        pair = (source_lang, target_lang)

        if backend == "nllb":
            if self._nllb is None:
                raise ValueError("NLLB backend is not configured (set TRANSLATION_MODEL).")
            if pair not in NLLB_PAIRS:
                raise ValueError(f"Pair {source_lang}→{target_lang} not supported by NLLB.")
            return self._load_nllb().translate(text, source_lang, target_lang), self._nllb.model_id

        if backend == "opus-mt":
            if pair not in OPUS_MT_PAIRS:
                raise ValueError(f"Pair {source_lang}→{target_lang} not supported by OPUS-MT.")
            opus = self._get_opus(source_lang, target_lang)
            return opus.translate(text), opus.model_id

        # Default: NLLB if configured, OPUS-MT otherwise
        if self._nllb is not None and not self._nllb_failed:
            try:
                return self._load_nllb().translate(text, source_lang, target_lang), self._nllb.model_id
            except Exception:
                logger.exception("NLLB failed, falling back to OPUS-MT")
                self._nllb_failed = True

        opus = self._get_opus(source_lang, target_lang)
        return opus.translate(text), opus.model_id

    @property
    def loaded_models(self) -> list[dict]:
        result = []
        if self._nllb and self._nllb.is_loaded:
            result.append({
                "backend": "nllb",
                "model_id": self._nllb.model_id,
                "device": self._nllb.device,
                "pairs": "all",
            })
        for (src, tgt), t in self._opus.items():
            result.append({
                "backend": "opus-mt",
                "pair": f"{src}-{tgt}",
                "model_id": t.model_id,
                "device": t.device,
            })
        return result