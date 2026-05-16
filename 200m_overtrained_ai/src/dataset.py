"""
Dataset streaming optimizat din CulturaX Romanian:
 - Tokenizare in BATCH (512 texte odata) -> 10x mai rapid vs. one-by-one
 - PrefetchBuffer pe thread separat -> GPU nu asteapta datele niciodata
 - Packing fara padding -> 100% eficienta a contextului
"""
import os
import sys
import threading
import queue as queue_module
from typing import Iterator, Dict, List

from datasets import load_dataset, IterableDataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import BytesTracker

# Dimensiunea batch-ului de tokenizare (texte per apel)
TOKENIZER_BATCH_SIZE = 512
# Numarul de chunk-uri pregatite in avans in buffer
PREFETCH_BUFFER_SIZE = 300


# ─── Prefetch Buffer ─────────────────────────────────────────────────────────

class PrefetchBuffer:
    """
    Ruleaza generatorul de date pe un thread de fundal, astfel incat
    GPU-ul nu va "stea" niciodata sa astepte urmatorul batch.
    """
    def __init__(self, source_iterable, buffer_size: int = PREFETCH_BUFFER_SIZE):
        self._queue = queue_module.Queue(maxsize=buffer_size)
        self._source = source_iterable
        self._sentinel = object()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self):
        try:
            for item in self._source:
                self._queue.put(item)
        except Exception as e:
            print(f"[PrefetchBuffer] Eroare producer: {e}")
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item


# ─── Streaming text ───────────────────────────────────────────────────────────

def _text_stream(language: str, hf_token: str = None, examples_to_skip: int = 0):
    dataset = load_dataset(
        "uonlp/CulturaX",
        language,
        split="train",
        streaming=True,
        token=hf_token,
    )
    if examples_to_skip > 0:
        dataset = dataset.skip(examples_to_skip)
    for item in dataset:
        text = item.get("text", "")
        if text and text.strip():
            yield text


# ─── Batch tokenizare + packing ───────────────────────────────────────────────

def _pack_tokens_batched(
    text_iter: Iterator[str],
    tokenizer: PreTrainedTokenizerFast,
    context_length: int,
    max_bytes: int,
    start_bytes: int = 0,
) -> Iterator[Dict[str, List[int]]]:
    """
    Tokenizeaza texte IN BATCH (512 odata) si le impacheteaza in
    chunk-uri de context_length tokens. Mult mai rapid decat one-by-one.
    """
    tracker = BytesTracker(max_bytes=max_bytes, start_bytes=start_bytes)
    buffer: List[int] = []
    bos = tokenizer.bos_token_id or 1
    eos = tokenizer.eos_token_id or 2

    text_batch: List[str] = []

    def flush_batch_to_buffer(texts: List[str]) -> List[int]:
        """Tokenizeaza un batch de texte si returneaza lista de tokens."""
        if not texts:
            return []
        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        result = []
        for ids in encodings:
            result.extend([bos] + ids + [eos])
        return result

    def yield_full_chunks():
        nonlocal buffer
        while len(buffer) >= context_length:
            chunk = buffer[:context_length]
            buffer = buffer[context_length:]
            yield {
                "input_ids": chunk,
                "labels": chunk,
                "attention_mask": [1] * context_length,
            }

    for text in text_iter:
        if not tracker.add(text):
            break
        text_batch.append(text)

        if len(text_batch) >= TOKENIZER_BATCH_SIZE:
            buffer.extend(flush_batch_to_buffer(text_batch))
            text_batch = []
            yield from yield_full_chunks()

    # Flush texte ramase
    if text_batch:
        buffer.extend(flush_batch_to_buffer(text_batch))

    yield from yield_full_chunks()

    # Ultimul chunk partial (daca e suficient de lung)
    if len(buffer) >= context_length // 2:
        pad_id = tokenizer.pad_token_id or 0
        chunk = buffer + [pad_id] * (context_length - len(buffer))
        chunk = chunk[:context_length]
        yield {
            "input_ids": chunk,
            "labels": chunk,
            "attention_mask": [1 if t != pad_id else 0 for t in chunk],
        }


# ─── Build dataset ────────────────────────────────────────────────────────────

def build_streaming_dataset(
    cfg: dict,
    tokenizer: PreTrainedTokenizerFast,
    examples_to_skip: int = 0,
    bytes_already_seen: int = 0,
    hf_token: str = None,
) -> IterableDataset:
    """
    Construieste IterableDataset optimizat cu:
    - Batch tokenizare (512 texte/apel)
    - Prefetch pe thread separat (300 chunk-uri in avans)
    """
    language = cfg["language"]
    max_bytes = cfg["max_train_bytes"]
    context_length = cfg.get("context_length", 2048)
    prefetch_size = cfg.get("prefetch_buffer_size", PREFETCH_BUFFER_SIZE)

    def gen():
        text_iter = _text_stream(language, hf_token, examples_to_skip)
        raw_gen = _pack_tokens_batched(
            text_iter, tokenizer, context_length, max_bytes, bytes_already_seen
        )
        # Prefetch pe thread de fundal
        prefetched = PrefetchBuffer(raw_gen, buffer_size=prefetch_size)
        yield from prefetched

    return IterableDataset.from_generator(gen)
