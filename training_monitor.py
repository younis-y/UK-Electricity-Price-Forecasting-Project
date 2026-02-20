#!/usr/bin/env python3
"""
training_monitor.py — Live terminal dashboard for model training.

Usage:
    python training_monitor.py

Opens a Rich Live display that tail-reads data/training_log.jsonl every 0.5 s
and renders pipeline stage, active model progress, completed models table, and
a mini loss chart.
"""

import json
import time
import sys
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.console import Console, Group
    from rich.bar import Bar
    from rich import box
except ImportError:
    print("ERROR: rich is required.  pip install rich")
    sys.exit(1)

LOG_PATH = Path(__file__).parent / "data" / "training_log.jsonl"

# ── State ────────────────────────────────────────────────────

session_info: dict = {}
current_stage: str = "Waiting for session…"
active_model: dict | None = None          # last model_start without a model_done
active_epochs: list[dict] = []            # epoch / xgb_round events for active model
completed: list[dict] = []                # model_done events
session_end_ts: float | None = None       # timestamp when session finished (None = still running)
_lines_read: int = 0
_model_categories: dict[str, str] = {}    # name → category (populated from model_start events)

# Fallback category inference from model_type
_TYPE_TO_CATEGORY = {
    "sklearn": "Classical ML",
    "xgboost": "Classical ML",
    "pytorch": "Deep Learning",
    "validation": "Validation",
}

# Category display colours
_CATEGORY_STYLES = {
    "Classical ML":         "bold blue",
    "Deep Learning":        "bold red",
    "Statistical Baseline": "bold white",
    "Validation":           "bold cyan",
}


def _get_category(ev: dict) -> str:
    """Extract category from event, falling back to model_type inference."""
    cat = ev.get("category", "")
    if cat:
        return cat
    name = ev.get("name", "")
    if name in _model_categories:
        return _model_categories[name]
    mtype = ev.get("model_type", "")
    return _TYPE_TO_CATEGORY.get(mtype, "")


def _reset():
    global session_info, current_stage, active_model, active_epochs, completed, session_end_ts, _lines_read, _model_categories
    session_info = {}
    current_stage = "Waiting for session…"
    active_model = None
    active_epochs = []
    completed = []
    session_end_ts = None
    _lines_read = 0
    _model_categories = {}


def _ingest():
    """Read new lines from the log file and update state."""
    global session_info, current_stage, active_model, active_epochs, completed, session_end_ts, _lines_read

    if not LOG_PATH.exists():
        return

    with open(LOG_PATH) as f:
        lines = f.readlines()

    # Detect truncation (new session)
    if len(lines) < _lines_read:
        _reset()

    new_lines = lines[_lines_read:]
    _lines_read = len(lines)

    for raw in new_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue

        kind = ev.get("event")

        if kind == "session_start":
            _reset()
            _lines_read = len(lines)  # keep cursor
            session_info = ev

        elif kind == "session_end":
            session_end_ts = ev.get("ts", time.time())
            current_stage = "Complete"
            active_model = None
            active_epochs = []

        elif kind == "stage":
            current_stage = ev["stage"]

        elif kind == "model_start":
            active_model = ev
            active_epochs = []
            # Store category mapping
            cat = _get_category(ev)
            if cat:
                _model_categories[ev["name"]] = cat

        elif kind in ("epoch", "xgb_round"):
            active_epochs.append(ev)

        elif kind == "model_done":
            # Enrich with category if not present
            if not ev.get("category") and ev.get("name") in _model_categories:
                ev["category"] = _model_categories[ev["name"]]
            completed.append(ev)
            if active_model and active_model["name"] == ev["name"]:
                active_model = None
                active_epochs = []


# ── Rendering helpers ────────────────────────────────────────

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _get_cpu_temp() -> float | None:
    """Try to get CPU temperature via multi-method fallback chain.

    1. psutil.sensors_temperatures() — Linux
    2. osx-cpu-temp CLI (Homebrew, no sudo) — macOS
    3. sudo -n powermetrics (only if sudo is cached) — macOS
    """
    # 1) psutil (Linux) — guard with hasattr since macOS lacks sensors_temperatures
    if psutil and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("coretemp", "cpu_thermal", "cpu-thermal", "k10temp"):
                    if name in temps and temps[name]:
                        return temps[name][0].current
                first = list(temps.values())[0]
                if first:
                    return first[0].current
        except Exception:
            pass

    # 2) osx-cpu-temp (Homebrew package, no sudo required)
    try:
        result = subprocess.run(
            ["osx-cpu-temp"], capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            # Output like "78.3°C"
            temp_str = result.stdout.strip().replace("°C", "").replace("C", "").strip()
            return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, Exception):
        pass

    # 3) powermetrics (Apple Silicon, only runs if sudo is cached — no password prompt)
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "smc", "-i1", "-n1"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "CPU die temperature" in line or "Die temperature" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        temp_str = parts[1].strip().replace("C", "").strip()
                        return float(temp_str)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception):
        pass
    return None


def _get_system_metrics() -> dict:
    """Gather CPU%, RAM usage, and temperatures."""
    metrics = {"cpu_pct": None, "ram_pct": None, "ram_used_gb": None,
               "ram_total_gb": None, "cpu_temp": None}
    if not psutil:
        return metrics
    try:
        metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        metrics["ram_pct"] = mem.percent
        metrics["ram_used_gb"] = mem.used / (1024 ** 3)
        metrics["ram_total_gb"] = mem.total / (1024 ** 3)
    except Exception:
        pass
    metrics["cpu_temp"] = _get_cpu_temp()
    return metrics


# Kick off non-blocking CPU measurement so first call isn't 0
if psutil:
    psutil.cpu_percent(interval=None)


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a mini sparkline string from a list of floats."""
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    rng = hi - lo if hi != lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in recent)


def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s2 = divmod(s, 60)
    return f"{int(m)}m {s2:.0f}s"


def _build_header() -> Panel:
    parts: list = []
    if session_info:
        ts = datetime.fromtimestamp(session_info.get("ts", 0)).strftime("%Y-%m-%d %H:%M:%S")
        device = session_info.get("device", "?")
        plat = session_info.get("platform", "")

        # Freeze timer at session end, otherwise keep ticking
        if session_end_ts is not None:
            elapsed_s = session_end_ts - session_info.get("ts", session_end_ts)
        else:
            elapsed_s = time.time() - session_info.get("ts", time.time())
        hrs, rem = divmod(int(elapsed_s), 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        status_parts = [
            ("UK Electricity Price Prediction", "bold cyan"),
            "  |  ",
            ("Started: ", "dim"), (ts, "green"),
            "  |  ",
            ("Device: ", "dim"), (device.upper(), "bold yellow"),
            "  |  ",
            (plat, "dim"),
            "  |  ",
        ]

        if session_end_ts is not None:
            status_parts += [("Total: ", "dim"), (elapsed_str, "bold green"), ("  COMPLETE", "bold green")]
        else:
            status_parts += [("Elapsed: ", "dim"), (elapsed_str, "bold white")]

        parts.append(Text.assemble(*status_parts))
    else:
        parts.append(Text("Waiting for training session to start…", style="dim italic"))

    # System metrics line
    m = _get_system_metrics()
    metric_parts: list = []

    if m["cpu_pct"] is not None:
        cpu_val = m["cpu_pct"]
        cpu_style = "bold green" if cpu_val < 50 else "bold yellow" if cpu_val < 80 else "bold red"
        metric_parts.append(("CPU: ", "dim"))
        metric_parts.append((f"{cpu_val:.1f}%", cpu_style))

    if m["cpu_temp"] is not None:
        temp = m["cpu_temp"]
        temp_style = "bold green" if temp < 60 else "bold yellow" if temp < 80 else "bold red"
        metric_parts.append(("  Temp: ", "dim"))
        metric_parts.append((f"{temp:.0f}°C", temp_style))
    else:
        metric_parts.append(("  Temp: ", "dim"))
        metric_parts.append(("N/A", "dim italic"))

    if m["ram_pct"] is not None:
        ram_style = "bold green" if m["ram_pct"] < 60 else "bold yellow" if m["ram_pct"] < 85 else "bold red"
        metric_parts.append(("  RAM: ", "dim"))
        metric_parts.append((f"{m['ram_used_gb']:.1f}/{m['ram_total_gb']:.0f}GB", ram_style))
        metric_parts.append((f" ({m['ram_pct']:.0f}%)", ram_style))

    if metric_parts:
        parts.append(Text.assemble(*metric_parts))

    return Panel(Group(*parts), title="[bold]Training Monitor[/bold]", border_style="bright_blue")


def _build_stage() -> Panel:
    if session_end_ts is not None:
        style = "bold white on green"
        border = "green"
        label = "  All training complete"
    else:
        style = "bold white on blue"
        border = "blue"
        label = f"  {current_stage}"
    return Panel(
        Text(label, style=style),
        title="[bold]Pipeline Stage[/bold]",
        border_style=border,
        expand=True,
    )


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_idx = 0


def _loss_delta(values: list[float]) -> tuple[str, str]:
    """Return (arrow_symbol, style) showing loss trend from the last two values."""
    if len(values) < 2:
        return ("", "dim")
    prev, curr = values[-2], values[-1]
    if prev == 0:
        return ("", "dim")
    pct = (curr - prev) / abs(prev) * 100
    if pct < -1:
        return (f" ▼{abs(pct):.1f}%", "bold green")
    elif pct > 1:
        return (f" ▲{abs(pct):.1f}%", "bold red")
    else:
        return (f" ◆{abs(pct):.1f}%", "yellow")


def _eta_str(current: int, total: int, elapsed_s: float) -> str:
    """Estimate time remaining based on average speed."""
    if current <= 0 or elapsed_s <= 0:
        return "calculating…"
    speed = elapsed_s / current
    remaining = (total - current) * speed
    if remaining < 60:
        return f"~{remaining:.0f}s"
    m, s = divmod(remaining, 60)
    return f"~{int(m)}m {s:.0f}s"


def _patience_bar(current: int, maximum: int, width: int = 15) -> Text:
    """Visual patience countdown bar: filled = used, empty = remaining."""
    filled = min(current, maximum)
    ratio = filled / maximum if maximum > 0 else 0
    n_filled = int(ratio * width)
    n_empty = width - n_filled
    if ratio > 0.8:
        style = "bold red"
    elif ratio > 0.5:
        style = "yellow"
    else:
        style = "green"
    return Text.assemble(
        ("▕", "dim"),
        ("█" * n_filled, style),
        ("░" * n_empty, "dim"),
        ("▏", "dim"),
        (f" {filled}/{maximum}", style),
    )


def _build_active() -> Panel:
    global _spinner_idx

    if not active_model:
        if session_end_ts is not None:
            if completed:
                best = max(completed, key=lambda m: m.get("r2", 0))
                total_dur = sum(m.get("duration_s", 0) for m in completed)
                return Panel(
                    Group(
                        Text.assemble(
                            ("✓ All models finished", "bold green"),
                            ("  |  Best: ", "dim"), (best["name"], "bold green"),
                            ("  R²=", "dim"), (f"{best.get('r2', 0):.4f}", "bold green"),
                            ("  |  ", "dim"),
                            (_fmt_duration(total_dur), "bold white"),
                            (" total  ", "dim"),
                            (f"({len(completed)} models)", "dim"),
                        ),
                    ),
                    title="[bold]Active Model[/bold]", border_style="green",
                )
            return Panel(Text("All training complete", style="bold green"),
                         title="[bold]Active Model[/bold]", border_style="green")
        return Panel(Text("No model currently training", style="dim"),
                     title="[bold]Active Model[/bold]", border_style="yellow")

    name = active_model["name"]
    mtype = active_model.get("model_type", "")
    hp = active_model.get("hyperparams", {})
    cat = _get_category(active_model)
    cat_style = _CATEGORY_STYLES.get(cat, "dim")
    start_ts = active_model.get("ts", time.time())
    elapsed = time.time() - start_ts

    _spinner_idx = (_spinner_idx + 1) % len(_SPINNER_FRAMES)
    spinner = _SPINNER_FRAMES[_spinner_idx]

    parts: list = []

    # ── Row 1: spinner + model name + category + elapsed ──
    row1_parts: list = [
        (f" {spinner} ", "bold yellow"),
        (name, "bold green"),
    ]
    if cat:
        row1_parts += [("  ", ""), ("⟨", "dim"), (cat, cat_style), ("⟩", "dim")]
    else:
        row1_parts += [("  ", ""), ("⟨", "dim"), (mtype, "cyan"), ("⟩", "dim")]
    row1_parts += [("  ", ""), ("⏱ ", "dim"), (_fmt_duration(elapsed), "yellow")]
    parts.append(Text.assemble(*row1_parts))

    # ── Row 1b: key hyperparams summary (DL models) ──
    info_bits: list = []
    if hp.get("n_params"):
        n = hp["n_params"]
        info_bits.append(f"{n / 1000:.0f}K params" if n < 1_000_000 else f"{n / 1_000_000:.1f}M params")
    if hp.get("max_lr"):
        info_bits.append(f"max_lr={hp['max_lr']}")
    elif hp.get("learning_rate"):
        info_bits.append(f"lr={hp['learning_rate']}")
    if hp.get("window"):
        info_bits.append(f"window={hp['window']}h")
    if info_bits:
        parts.append(Text.assemble(("  ", ""), ("  |  ".join(info_bits), "dim")))

    if not active_epochs:
        parts.append(Text("  Initialising…", style="dim italic"))
        return Panel(Group(*parts), title="[bold]Active Model[/bold]", border_style="yellow")

    last = active_epochs[-1]
    kind = last.get("event")

    if kind == "epoch":
        ep = last["epoch"]
        mx = last["max_epochs"]
        pct = (ep / mx * 100) if mx else 0
        tl = last.get("train_loss", 0)
        vl = last.get("val_loss", 0)
        bvl = last.get("best_val_loss", 0)
        lr = last.get("lr", 0)
        pat = last.get("patience_counter", 0)
        max_patience = hp.get("patience", 40)

        speed_str = f"{elapsed / ep:.1f}s/ep" if ep > 0 else ""
        eta = _eta_str(ep, mx, elapsed)

        train_losses = [e.get("train_loss", 0) for e in active_epochs if e.get("event") == "epoch"]
        val_losses = [e.get("val_loss", 0) for e in active_epochs if e.get("event") == "epoch"]
        t_arrow, t_arrow_style = _loss_delta(train_losses)
        v_arrow, v_arrow_style = _loss_delta(val_losses)

        # ── Row 2: progress ──
        pbar_filled = int(pct / 100 * 30)
        pbar_empty = 30 - pbar_filled
        parts.append(Text.assemble(
            ("  ", ""),
            ("━" * pbar_filled, "bold cyan"),
            ("╺" if pbar_empty > 0 else "", "dim"),
            ("━" * max(0, pbar_empty - 1), "dim"),
            (f" {pct:.0f}%", "bold cyan"),
            ("  ", ""), (speed_str, "dim"),
            ("  ETA ", "dim"), (eta, "bold white"),
        ))

        # ── Row 3: losses with deltas ──
        parts.append(Text.assemble(
            ("  Train: ", "dim"), (f"{tl:.6f}", "white"), (t_arrow, t_arrow_style),
            ("  Val: ", "dim"), (f"{vl:.6f}", "white"), (v_arrow, v_arrow_style),
        ))

        # ── Row 4: best + reduction + LR ──
        imp_parts: list = [("  Best: ", "dim"), (f"{bvl:.6f}", "bold green")]
        if len(val_losses) > 1 and val_losses[0] > 0:
            improvement = (1 - bvl / val_losses[0]) * 100
            imp_parts += [("  ↓", "bold green"), (f"{improvement:.1f}%", "bold green")]
        imp_parts += [("  LR ", "dim"), (f"{lr:.2e}", "cyan")]
        parts.append(Text.assemble(*imp_parts))

        # ── Row 5: patience bar ──
        parts.append(Text.assemble(("  Patience ", "dim")))
        parts.append(_patience_bar(pat, max_patience))

        # ── Row 5 (conditional): overfitting warning ──
        if len(train_losses) > 5 and len(val_losses) > 5:
            gap = val_losses[-1] - train_losses[-1]
            gap_prev = val_losses[-5] - train_losses[-5]
            if gap > 0 and gap_prev > 0 and gap > gap_prev * 1.5:
                parts.append(Text("  ⚠ Train-val gap widening — possible overfitting", style="bold red"))

        # ── Row 6-7: sparklines ──
        parts.append(Text.assemble(
            ("  Train ▸ ", "dim"), (_sparkline(train_losses), "yellow"),
        ))
        parts.append(Text.assemble(
            ("  Val   ▸ ", "dim"), (_sparkline(val_losses), "magenta"),
        ))

    elif kind == "xgb_round":
        rd = last["round"]
        mx = last["n_rounds"]
        pct = (rd / mx * 100) if mx else 0
        tr = last.get("train_rmse", 0)
        vr = last.get("val_rmse", 0)

        speed_str = f"{rd / elapsed:.0f} rnd/s" if elapsed > 0 else ""
        eta = _eta_str(rd, mx, elapsed)

        train_vals = [e.get("train_rmse", 0) for e in active_epochs if e.get("event") == "xgb_round"]
        val_vals = [e.get("val_rmse", 0) for e in active_epochs if e.get("event") == "xgb_round"]
        t_arrow, t_arrow_style = _loss_delta(train_vals)
        v_arrow, v_arrow_style = _loss_delta(val_vals)

        # ── Row 2: progress ──
        pbar_filled = int(pct / 100 * 30)
        pbar_empty = 30 - pbar_filled
        parts.append(Text.assemble(
            ("  ", ""),
            ("━" * pbar_filled, "bold cyan"),
            ("╺" if pbar_empty > 0 else "", "dim"),
            ("━" * max(0, pbar_empty - 1), "dim"),
            (f" {pct:.0f}%", "bold cyan"),
            ("  ", ""), (speed_str, "dim"),
            ("  ETA ", "dim"), (eta, "bold white"),
        ))

        # ── Row 3-4: RMSE with deltas ──
        parts.append(Text.assemble(
            ("  Train: ", "dim"), (f"{tr:.4f}", "white"), (t_arrow, t_arrow_style),
            ("  Val: ", "dim"), (f"{vr:.4f}", "white"), (v_arrow, v_arrow_style),
        ))
        if len(val_vals) > 1 and val_vals[0] > 0:
            improvement = (1 - vr / val_vals[0]) * 100
            parts.append(Text.assemble(
                ("  Reduced ", "dim"), (f"{improvement:.1f}%", "bold green"),
            ))

        # ── Row 4-5: sparklines ──
        parts.append(Text.assemble(
            ("  Train ▸ ", "dim"), (_sparkline(train_vals), "yellow"),
        ))
        parts.append(Text.assemble(
            ("  Val   ▸ ", "dim"), (_sparkline(val_vals), "magenta"),
        ))

    return Panel(Group(*parts), title="[bold]Active Model[/bold]", border_style="yellow")


def _build_completed() -> Panel:
    if not completed:
        return Panel(Text("No models completed yet", style="dim"), title="[bold]Completed Models[/bold]", border_style="green")

    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="bold", no_wrap=True, min_width=18)
    table.add_column("Category", no_wrap=True, min_width=12)
    table.add_column("R²", justify="right", min_width=8, no_wrap=True)
    table.add_column("RMSE", justify="right", min_width=7, no_wrap=True)
    table.add_column("MAE", justify="right", min_width=7, no_wrap=True)
    table.add_column("Time", justify="right", min_width=7, no_wrap=True)

    sorted_models = sorted(completed, key=lambda m: m.get("r2", 0), reverse=True)
    for i, m in enumerate(sorted_models, 1):
        r2 = m.get("r2", 0)
        r2_style = "bold green" if r2 > 0.9 else "green" if r2 > 0.8 else "yellow" if r2 > 0.5 else "red"
        cat = m.get("category", _model_categories.get(m.get("name", ""), ""))
        cat_style = _CATEGORY_STYLES.get(cat, "dim")
        table.add_row(
            str(i),
            m["name"],
            Text(cat, style=cat_style) if cat else Text("—", style="dim"),
            Text(f"{r2:.4f}", style=r2_style),
            f"{m.get('rmse', 0):.2f}",
            f"{m.get('mae', 0):.2f}",
            _fmt_duration(m.get("duration_s", 0)),
        )

    return Panel(table, title=f"[bold]Completed Models ({len(completed)})[/bold]", border_style="green")


def _build_loss_chart() -> Panel:
    """Mini bar chart of last 20 val losses for the active model."""
    vals = []
    label = "Val Loss"
    if active_epochs:
        kind = active_epochs[-1].get("event")
        if kind == "epoch":
            vals = [e.get("val_loss", 0) for e in active_epochs if e.get("event") == "epoch"]
            label = "Val Loss (MSE)"
        elif kind == "xgb_round":
            vals = [e.get("val_rmse", 0) for e in active_epochs if e.get("event") == "xgb_round"]
            label = "Val RMSE"

    if not vals:
        return Panel(Text("No iterative data yet", style="dim"), title="[bold]Loss Trend[/bold]", border_style="magenta", height=8)

    recent = vals[-30:]
    lo, hi = min(recent), max(recent)
    rng = hi - lo if hi != lo else 1.0

    lines: list[Text] = []
    lines.append(Text(f"{label}: {recent[-1]:.6f}  (min: {lo:.6f}, max: {hi:.6f})", style="dim"))

    # 5-row bar chart
    chart_height = 5
    for row in range(chart_height, 0, -1):
        threshold = lo + (row / chart_height) * rng
        chars = []
        for v in recent:
            if v >= threshold:
                chars.append("█")
            else:
                chars.append(" ")
        lines.append(Text("".join(chars), style="magenta"))

    return Panel(Group(*lines), title="[bold]Loss Trend[/bold]", border_style="magenta", height=8)


def build_dashboard() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="stage", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["left"].split_column(
        Layout(name="active", minimum_size=12),
        Layout(name="loss", size=10),
    )

    layout["header"].update(_build_header())
    layout["stage"].update(_build_stage())
    layout["active"].update(_build_active())
    layout["loss"].update(_build_loss_chart())
    layout["right"].update(_build_completed())
    return layout


# ── Main ─────────────────────────────────────────────────────

def main():
    console = Console()
    console.clear()

    # Always start fresh — replay the full log from scratch
    _reset()

    console.print("[bold cyan]Training Monitor[/bold cyan] — watching [yellow]data/training_log.jsonl[/yellow]")
    console.print("Press [bold]Ctrl+C[/bold] to quit.\n")

    if not LOG_PATH.exists():
        console.print("[dim]Log file not found yet. Will appear when training starts…[/dim]\n")

    # Initial ingest to catch up with any existing log
    _ingest()

    try:
        with Live(build_dashboard(), console=console, refresh_per_second=2, screen=True) as live:
            while True:
                _ingest()
                live.update(build_dashboard())
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[bold]Monitor stopped.[/bold]")


if __name__ == "__main__":
    main()
