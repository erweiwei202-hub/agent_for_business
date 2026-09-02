#!/usr/bin/env python3
"""Plot training loss from a Hugging Face checkpoint directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_trainer_state(input_dir: Path) -> Path:
    direct = input_dir / "trainer_state.json"
    if direct.exists():
        return direct

    candidates = sorted(
        input_dir.glob("**/trainer_state.json"),
        key=lambda p: (checkpoint_step(p.parent), p.stat().st_mtime),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found under: {input_dir}\n"
            "Pass either a checkpoint folder like checkpoint-294, or a parent folder containing checkpoints."
        )
    return candidates[-1]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(mean(values[start : index + 1]))
    return averaged


def load_history(trainer_state_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with trainer_state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    history = state.get("log_history") or []
    if not isinstance(history, list):
        raise ValueError(f"Invalid log_history in {trainer_state_path}")
    return history, state


def collect_series(history: list[dict[str, Any]], metric: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for item in history:
        if metric not in item or "step" not in item:
            continue
        value = item[metric]
        if value is None:
            continue
        steps.append(int(item["step"]))
        values.append(float(value))
    return steps, values


def default_output_path(input_dir: Path, trainer_state_path: Path) -> Path:
    if (input_dir / "trainer_state.json").exists():
        return input_dir / "loss.png"
    return trainer_state_path.parent / "loss.png"


def plot_loss(
    input_dir: Path,
    output_path: Path | None,
    smooth_window: int,
    title: str | None,
    dpi: int,
) -> Path:
    trainer_state_path = find_trainer_state(input_dir)
    history, state = load_history(trainer_state_path)

    train_steps, train_loss = collect_series(history, "loss")
    eval_steps, eval_loss = collect_series(history, "eval_loss")

    if not train_loss and not eval_loss:
        raise ValueError(f"No loss or eval_loss entries found in {trainer_state_path}")

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.6))

    if train_loss:
        ax.plot(
            train_steps,
            train_loss,
            color="#2563eb",
            linewidth=1.2,
            alpha=0.35 if smooth_window > 1 else 1.0,
            label="train loss",
        )
        if smooth_window > 1:
            ax.plot(
                train_steps,
                moving_average(train_loss, smooth_window),
                color="#1d4ed8",
                linewidth=2.2,
                label=f"train loss MA({smooth_window})",
            )

    if eval_loss:
        ax.plot(
            eval_steps,
            eval_loss,
            color="#dc2626",
            marker="o",
            markersize=4,
            linewidth=2.0,
            label="eval loss",
        )

    final_step = state.get("global_step")
    final_epoch = state.get("epoch")
    title_parts = [title or "Training Loss"]
    subtitle = []
    if final_step is not None:
        subtitle.append(f"step {final_step}")
    if final_epoch is not None:
        subtitle.append(f"epoch {final_epoch:g}")
    if subtitle:
        title_parts.append(" / ".join(subtitle))

    ax.set_title(" — ".join(title_parts), fontsize=14, pad=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.margins(x=0.01)
    ax.grid(True, color="#e5e7eb")
    fig.tight_layout()

    output = output_path or default_output_path(input_dir, trainer_state_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss from a Hugging Face trainer_state.json in a checkpoint folder."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Checkpoint folder, or parent folder containing checkpoint-*/trainer_state.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to loss.png beside the detected trainer_state.json.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Moving-average window for train loss. Use 1 to disable smoothing. Default: 10.",
    )
    parser.add_argument("--title", default=None, help="Custom chart title.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI. Default: 160.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else None
    saved_to = plot_loss(
        input_dir=checkpoint_dir,
        output_path=output,
        smooth_window=args.smooth_window,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Saved loss plot to: {saved_to}")


if __name__ == "__main__":
    main()
