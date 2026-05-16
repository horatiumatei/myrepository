"""
Initializare model Llama (~210M parametri) cu Flash Attention 2.
Fallback automat la SDPA daca flash-attn nu e instalat.
"""
import os
import sys

from transformers import LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import setup_logging

logger = setup_logging()


def _detect_attn_implementation() -> str:
    """Alege cea mai rapida implementare de attention disponibila."""
    try:
        import flash_attn
        logger.info(f"✓ Flash Attention 2 detectat (v{flash_attn.__version__}) — viteza maxima!")
        return "flash_attention_2"
    except ImportError:
        logger.warning("Flash Attention 2 nu e instalat. Folosesc SDPA (PyTorch native).")
        logger.warning("Pentru viteza maxima: pip install flash-attn --no-build-isolation")
        return "sdpa"


def build_model_config(model_cfg: dict, tokenizer: PreTrainedTokenizerFast) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]),
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg["rms_norm_eps"],
        initializer_range=model_cfg["initializer_range"],
        use_cache=False,  # Dezactivat in antrenare
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return config


def build_model(model_cfg: dict, tokenizer: PreTrainedTokenizerFast) -> LlamaForCausalLM:
    config = build_model_config(model_cfg, tokenizer)
    attn_impl = _detect_attn_implementation()

    model = LlamaForCausalLM._from_config(
        config,
        attn_implementation=attn_impl,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Llama initializat:")
    logger.info(f"  Total parametri    : {total_params / 1e6:.1f}M")
    logger.info(f"  Antrenabili        : {trainable_params / 1e6:.1f}M")
    logger.info(f"  Attention backend  : {attn_impl}")
    logger.info(f"  Config             : {config.num_hidden_layers}L / {config.hidden_size}D / {config.num_attention_heads}H (GQA kv={config.num_key_value_heads})")

    return model


def load_model_from_checkpoint(checkpoint_path: str) -> LlamaForCausalLM:
    attn_impl = _detect_attn_implementation()
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path,
        attn_implementation=attn_impl,
    )
    logger.info(f"Model incarcat din: {checkpoint_path} (attn={attn_impl})")
    return model
