"""
Antreneaza un tokenizer ByteLevel BPE pe 2GB din CulturaX Romanian.
Ruleaza inainte de antrenarea modelului.

Utilizare:
    python src/tokenizer_train.py --config config/train_config.yaml
"""
import os
import sys
import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import load_config, setup_logging, BytesTracker

logger = setup_logging()

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def culturax_ro_text_iterator(sample_bytes: int, hf_token: str = None):
    """Generator de texte romanesti din CulturaX pana la sample_bytes."""
    logger.info(f"Streaming CulturaX (ro) pentru tokenizer — limita: {sample_bytes / 1e9:.1f} GB")
    dataset = load_dataset(
        "uonlp/CulturaX",
        "ro",
        split="train",
        streaming=True,
        token=hf_token,
    )
    tracker = BytesTracker(max_bytes=sample_bytes)
    for item in dataset:
        text = item["text"]
        if not text or not text.strip():
            continue
        yield text
        if not tracker.add(text):
            logger.info(f"Atins {tracker.gb_max:.1f} GB pentru tokenizer. Stop.")
            break


def train_tokenizer(cfg: dict, hf_token: str = None):
    tokenizer_path = cfg["tokenizer_path"]
    vocab_size = cfg["tokenizer_vocab_size"]
    sample_bytes = cfg["tokenizer_sample_bytes"]

    Path(tokenizer_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Antrenare tokenizer BPE {vocab_size // 1000}k vocab pe {sample_bytes / 1e9:.1f} GB date...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        culturax_ro_text_iterator(sample_bytes, hf_token),
        vocab_size=vocab_size,
        min_frequency=3,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Salveaza vocabularul raw
    tokenizer.save_model(tokenizer_path)
    logger.info(f"Vocabular raw salvat in: {tokenizer_path}")

    # Wrap cu PreTrainedTokenizerFast pentru compatibilitate HuggingFace
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=None,
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    fast_tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer HuggingFace salvat in: {tokenizer_path}")
    logger.info(f"Vocab size final: {fast_tokenizer.vocab_size}")

    return fast_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Antrenare tokenizer BPE pentru CulturaX Romanian")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--hf_token", default=None, help="HuggingFace API token")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer_path = cfg["tokenizer_path"]

    if Path(tokenizer_path).exists() and (Path(tokenizer_path) / "tokenizer.json").exists():
        logger.info(f"Tokenizer exista deja la: {tokenizer_path}. Skip antrenare.")
        logger.info("Sterge folderul tokenizer/ daca vrei sa reantrenezi.")
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("Nu ai setat HF_TOKEN. CulturaX necesita autentificare HuggingFace.")
        logger.warning("Seteaza: export HF_TOKEN=hf_xxxx  sau  --hf_token hf_xxxx")

    train_tokenizer(cfg, hf_token)
    logger.info("✓ Tokenizer antrenat cu succes!")


if __name__ == "__main__":
    main()
