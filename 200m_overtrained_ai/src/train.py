"""
Script principal de antrenare — 200M LLM pe CulturaX Romanian.
Optimizari maximale pentru L4 GPU:
  - Flash Attention 2 (auto-detect, fallback SDPA)
  - torch.compile(mode='max-autotune')
  - Gradient checkpointing cu use_reentrant=False
  - TF32 matmuls
  - Fused AdamW
  - Tokenizer parallelism
  - Prefetch buffer (thread separat)
  - Batch tokenizare 512 texte/apel

Utilizare:
    python src/train.py --model_config config/model_config.yaml \\
                        --train_config config/train_config.yaml \\
                        --hf_token hf_xxxxx
"""
import os
import sys
import argparse
from pathlib import Path

# ── Setari env inainte de orice import torch ──────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

import torch
import torch.backends.cuda
import torch.backends.cudnn

from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
    default_data_collator,
    TrainerCallback,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import (
    load_both_configs,
    setup_logging,
    find_latest_checkpoint,
    load_training_state,
    save_training_state,
)
from src.model import build_model, load_model_from_checkpoint
from src.dataset import build_streaming_dataset

logger = setup_logging()


# ─── Callback checkpoint ──────────────────────────────────────────────────────

class SaveStateCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: str, tracker_ref: dict):
        self.checkpoint_dir = checkpoint_dir
        self.tracker = tracker_ref

    def on_save(self, args, state, control, **kwargs):
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint-{state.global_step}")
        save_training_state({
            "global_step": state.global_step,
            "bytes_seen": self.tracker.get("bytes", 0),
            "examples_seen": self.tracker.get("examples", 0),
            "epoch": state.epoch,
        }, ckpt_path)
        logger.info(
            f"State salvat @ step {state.global_step} | "
            f"{self.tracker.get('bytes', 0) / 1e9:.2f} GB procesate"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="config/model_config.yaml")
    parser.add_argument("--train_config", default="config/train_config.yaml")
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    model_cfg, train_cfg = load_both_configs(args.model_config, args.train_config)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    checkpoint_dir = train_cfg["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Optimizari globale PyTorch ────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    logger.info("✓ TF32 si float32 high precision activate.")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer_path = train_cfg["tokenizer_path"]
    if not (Path(tokenizer_path) / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"Tokenizer nu gasit la: {tokenizer_path}\n"
            "Ruleaza mai intai: python src/tokenizer_train.py"
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    logger.info(f"✓ Tokenizer incarcat: vocab_size={tokenizer.vocab_size}")

    # ── 2. Resume ─────────────────────────────────────────────────────────────
    examples_to_skip = 0
    bytes_already_seen = 0
    resume_checkpoint = None

    if train_cfg.get("resume_from_checkpoint", True):
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            resume_checkpoint = latest
            state = load_training_state(latest)
            if state:
                examples_to_skip = state.get("examples_seen", 0)
                bytes_already_seen = state.get("bytes_seen", 0)
                logger.info(
                    f"✓ Resume din {latest} | "
                    f"step={state.get('global_step',0)} | "
                    f"{bytes_already_seen/1e9:.2f} GB"
                )
        else:
            logger.info("Antrenare de la zero.")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    if resume_checkpoint:
        model = load_model_from_checkpoint(resume_checkpoint)
    else:
        model = build_model(model_cfg, tokenizer)

    # Gradient checkpointing — DEZACTIVAT default (210M incape in 24GB fara el)
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("✓ Gradient checkpointing activat.")
    else:
        logger.info("✓ Gradient checkpointing DEZACTIVAT → +25% throughput.")

    # torch.compile cu max-autotune + fullgraph pentru kernel fusion maxim
    if train_cfg.get("torch_compile", True):
        compile_mode = train_cfg.get("torch_compile_mode", "max-autotune")
        logger.info(f"Compilare model cu torch.compile(mode='{compile_mode}', fullgraph=True)...")
        logger.info("(Prima rulare va dura 5-10 min pentru compilare — normal!)")
        model = torch.compile(model, mode=compile_mode, fullgraph=True)
        logger.info(f"✓ torch.compile activat (mode={compile_mode}, fullgraph=True).")

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    tracker_ref = {"bytes": bytes_already_seen, "examples": examples_to_skip}
    train_dataset = build_streaming_dataset(
        cfg=train_cfg,
        tokenizer=tokenizer,
        examples_to_skip=examples_to_skip,
        bytes_already_seen=bytes_already_seen,
        hf_token=hf_token,
    )

    # ── 5. TrainingArguments ──────────────────────────────────────────────────
    use_bf16 = train_cfg["dtype"] == "bfloat16" and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and train_cfg["dtype"] == "float16"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_epsilon"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        dataloader_num_workers=0,       # streaming necesita 0
        dataloader_pin_memory=True,     # transfer rapid CPU->GPU
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        run_name=train_cfg.get("wandb_project", "200m_ro"),
        remove_unused_columns=False,
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        torch_compile=False,            # compilam manual cu max-autotune
        ddp_find_unused_parameters=False,
    )

    # ── 6. Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[SaveStateCallback(checkpoint_dir, tracker_ref)],
    )

    # ── 7. Start ──────────────────────────────────────────────────────────────
    eff_batch = train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    tokens_per_step = eff_batch * train_cfg.get("context_length", 2048)
    logger.info("=" * 60)
    logger.info("START ANTRENARE — 200M LLM Romanian")
    logger.info(f"  Steps            : {train_cfg['max_steps']:,}")
    logger.info(f"  Batch efectiv    : {eff_batch} seq")
    logger.info(f"  Tokens / step    : {tokens_per_step:,}")
    logger.info(f"  Total tokens     : ~{train_cfg['max_steps'] * tokens_per_step / 1e9:.2f}B")
    logger.info(f"  Precision        : {'bfloat16' if use_bf16 else 'float16' if use_fp16 else 'float32'}")
    logger.info(f"  Checkpoint dir   : {checkpoint_dir}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── 8. Salvare finala ─────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"✓ Model final salvat: {final_path}")


if __name__ == "__main__":
    main()
