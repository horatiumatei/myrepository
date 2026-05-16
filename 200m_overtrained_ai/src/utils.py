"""Utilități comune: logging, config loading, bytes tracking."""
import os
import yaml
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )
    return logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_both_configs(model_cfg_path: str, train_cfg_path: str):
    m = load_config(model_cfg_path)
    t = load_config(train_cfg_path)
    return m, t


def save_training_state(state: dict, checkpoint_dir: str):
    """Salveaza starea de antrenare (steps, bytes) in checkpoint dir."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_training_state(checkpoint_dir: str) -> Optional[dict]:
    """Incarca starea anterioara daca exista."""
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return None


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Gaseste cel mai recent checkpoint HuggingFace din dir."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


class BytesTracker:
    """Urmareste bytes procesati si opreste la limita."""
    def __init__(self, max_bytes: int, start_bytes: int = 0):
        self.max_bytes = max_bytes
        self.bytes_seen = start_bytes

    def add(self, text: str) -> bool:
        """Adauga text si returneaza False daca am depasit limita."""
        self.bytes_seen += len(text.encode("utf-8"))
        return self.bytes_seen < self.max_bytes

    @property
    def progress_pct(self) -> float:
        return 100.0 * self.bytes_seen / self.max_bytes

    @property
    def gb_seen(self) -> float:
        return self.bytes_seen / (1024 ** 3)

    @property
    def gb_max(self) -> float:
        return self.max_bytes / (1024 ** 3)
