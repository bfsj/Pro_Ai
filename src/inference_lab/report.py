"""Render measured results. No example or fabricated performance numbers."""

import json
from pathlib import Path


def plot_report(input_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(input_path).read_text(encoding="utf-8"))
    rows = report["summary"]
    if not rows:
        raise ValueError("No successful measurements to plot")
    batches = sorted({row["batch_size"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [
        ("output_tokens_per_second_median", "Output throughput (tokens/s)"),
        ("decode_step_ms_median", "Mean decode step (ms)"),
        ("peak_allocated_mib_median", "Peak allocated GPU memory (MiB)"),
    ]
    colors = plt.get_cmap("tab10")
    for index, batch in enumerate(batches):
        for cache in (False, True):
            selected = sorted((r for r in rows if r["batch_size"] == batch and r["use_cache"] == cache),
                              key=lambda r: r["input_length"])
            for ax, (metric, label) in zip(axes, metrics):
                valid = [r for r in selected if r[metric] is not None]
                if valid:
                    ax.plot([r["input_length"] for r in valid], [r[metric] for r in valid],
                            marker="o", linestyle="-" if cache else "--", color=colors(index % 10),
                            label=f"B={batch}, cache={'on' if cache else 'off'}")
                ax.set_xlabel("Input tokens per sequence")
                ax.set_ylabel(label)
                ax.grid(alpha=0.25)
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No measurements for this metric", ha="center", transform=ax.transAxes)
    env = report["environment"]
    fig.suptitle(f"{report['configuration']['model']} | {env.get('gpu', 'CPU')} | medians of repeated runs")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        fig.savefig(file, format="png", dpi=160)
    plt.close(fig)
    return path
