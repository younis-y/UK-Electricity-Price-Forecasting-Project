"""
training_logger.py — Lightweight JSON-lines logger for model training.

Writes events to data/training_log.jsonl. A separate terminal monitor
(training_monitor.py) tail-reads this file and renders a live dashboard.
"""

import json
import time
import platform
from pathlib import Path

LOG_PATH = Path(__file__).parent / "data" / "training_log.jsonl"


def _write(event: dict):
    """Append a single JSON line with a timestamp."""
    event["ts"] = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


# ── Session lifecycle ────────────────────────────────────────

def start_run():
    """Clear log and write a session header."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")  # truncate
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"
    _write({
        "event": "session_start",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
    })


def end_run():
    """Write a session-end marker so the monitor can freeze the timer."""
    _write({"event": "session_end"})


def log_stage(stage_name: str):
    """Mark a new pipeline stage (e.g. 'Feature Engineering', 'DL Training')."""
    _write({"event": "stage", "stage": stage_name})


# ── Model lifecycle ──────────────────────────────────────────

def log_model_start(name: str, model_type: str = "", hyperparams: dict | None = None,
                    category: str = ""):
    _write({
        "event": "model_start",
        "name": name,
        "model_type": model_type,
        "hyperparams": hyperparams or {},
        "category": category,
    })


def log_model_done(name: str, r2: float, rmse: float, mae: float, duration_s: float,
                   category: str = ""):
    _write({
        "event": "model_done",
        "name": name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "duration_s": duration_s,
        "category": category,
    })


# ── Iterative training progress ─────────────────────────────

def log_epoch(name: str, epoch: int, max_epochs: int,
              train_loss: float, val_loss: float,
              lr: float = 0.0, patience_counter: int = 0,
              best_val_loss: float = 0.0):
    _write({
        "event": "epoch",
        "name": name,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": lr,
        "patience_counter": patience_counter,
        "best_val_loss": best_val_loss,
    })


def log_xgb_round(name: str, round_: int, n_rounds: int,
                   train_rmse: float, val_rmse: float):
    _write({
        "event": "xgb_round",
        "name": name,
        "round": round_,
        "n_rounds": n_rounds,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    })
