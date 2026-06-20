"""Plot generation for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _prepare_output(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_quadrant_stacked_bars(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Generate stacked bar chart for required quadrant proportions."""
    out = _prepare_output(output_path)
    counts = metrics["quadrant_counts"]
    labels = [
        "Answerable & Correct",
        "Answerable & Wrong",
        "Unanswerable & Abstain",
        "Unanswerable & Assert",
    ]
    values = [
        counts["answerable_correct"],
        counts["answerable_wrong"],
        counts["unanswerable_abstain"],
        counts["unanswerable_assert"],
    ]
    colors = ["#2ca25f", "#de2d26", "#3182bd", "#ff8c00"]

    fig, ax = plt.subplots(figsize=(8, 4))
    left = 0
    total = max(1, sum(values))
    for label, value, color in zip(labels, values, colors):
        ax.barh(["Quadrants"], [value], left=[left], color=color, label=label)
        left += value
    ax.set_xlim(0, total)
    ax.set_xlabel("Count")
    ax.set_title("Quadrant Stacked Bar")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_coverage_risk_curve(curve: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Generate coverage-risk curve."""
    out = _prepare_output(output_path)
    coverage = [float(row["coverage"]) for row in curve]
    risk = [float(row["risk"]) for row in curve]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(coverage, risk, marker="o", linewidth=1.5, markersize=3, color="#1f77b4")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Answerable Risk-Coverage Curve")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_assert_abstain_curve(curve: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Generate unanswerable assert-abstain curve."""
    out = _prepare_output(output_path)
    assert_rate = [float(row["assert_rate"]) for row in curve]
    abstain_rate = [float(row["abstain_rate"]) for row in curve]
    thresholds = [float(row["threshold"]) for row in curve]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, assert_rate, marker="o", linewidth=1.5, markersize=3, color="#d94801", label="Assert")
    ax.plot(
        thresholds,
        abstain_rate,
        marker="s",
        linewidth=1.2,
        markersize=3,
        color="#3182bd",
        label="Abstain",
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Unanswerable Assert-Abstain Curve")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confidence_calibration(calibration: dict[str, Any], output_path: str | Path) -> Path:
    """Generate reliability-style confidence calibration plot."""
    out = _prepare_output(output_path)
    bins = calibration.get("bins", [])
    xs = [((b["bin_start"] + b["bin_end"]) / 2.0) for b in bins]
    ys = [float(b["empirical_accuracy"]) for b in bins]
    ws = [float(b["bin_end"] - b["bin_start"]) for b in bins]
    counts = [int(b["count"]) for b in bins]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.bar(xs, ys, width=ws, alpha=0.5, color="#31a354", edgecolor="black")
    for x, y, c in zip(xs, ys, counts):
        if c > 0:
            ax.text(x, y + 0.02, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(f"Calibration Plot (ECE={calibration.get('ece', 0.0):.3f})")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_all_plots(metrics: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Save all required plots to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quadrant_path = plot_quadrant_stacked_bars(
        metrics=metrics,
        output_path=out_dir / "quadrant_stacked_bar.png",
    )
    curve_path = plot_coverage_risk_curve(
        curve=metrics.get("answerable_risk_coverage_curve", metrics["coverage_risk_curve"]),
        output_path=out_dir / "coverage_risk_curve.png",
    )
    assert_abstain_path = plot_assert_abstain_curve(
        curve=metrics["unanswerable_assert_abstain_curve"],
        output_path=out_dir / "assert_abstain_curve.png",
    )
    calibration_path = plot_confidence_calibration(
        calibration=metrics["calibration"],
        output_path=out_dir / "confidence_calibration.png",
    )
    return {
        "quadrant_stacked_bar": str(quadrant_path),
        "coverage_risk_curve": str(curve_path),
        "assert_abstain_curve": str(assert_abstain_path),
        "confidence_calibration": str(calibration_path),
    }
