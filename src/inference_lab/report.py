"""Render measured cache/normalization variants without inventing data."""

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

    def cache_mode(row):
        return row.get("cache_mode", "dynamic" if row.get("use_cache") else "no_cache")

    batches = sorted({r["batch_size"] for r in rows})
    variants = sorted({(cache_mode(r), r.get("norm_backend", "torch")) for r in rows})
    modes = sorted({mode for mode, _ in variants})
    palette = plt.get_cmap("tab10")
    colors = {mode: palette(index) for index, mode in enumerate(modes)}
    fig, axes = plt.subplots(len(batches), 3, figsize=(16, 3.6 * len(batches) + 1),
                             squeeze=False)
    metrics = [
        ("output_tokens_per_second_median", "Output throughput (tokens/s)"),
        ("decode_step_ms_median", "Mean decode step (ms)"),
        ("peak_allocated_mib_median", "Peak allocated GPU memory (MiB)"),
    ]
    handles = {}
    for batch, row_axes in zip(batches, axes):
        for mode, norm in variants:
            selected = sorted((r for r in rows if r["batch_size"] == batch
                               and cache_mode(r) == mode
                               and r.get("norm_backend", "torch") == norm),
                              key=lambda r: r["input_length"])
            for ax, (metric, label) in zip(row_axes, metrics):
                valid = [r for r in selected if r.get(metric) is not None]
                if valid:
                    line, = ax.plot(
                        [r["input_length"] for r in valid], [r[metric] for r in valid],
                        marker="o" if norm == "torch" else "s", color=colors[mode],
                        linestyle="--" if norm == "torch" else "-", label=f"{mode}/{norm}",
                    )
                    handles[(mode, norm)] = line
                ax.set(xlabel="Input tokens per sequence", ylabel=label, title=f"Batch = {batch}")
                ax.grid(alpha=0.25)
        for ax in row_axes:
            if not ax.lines:
                ax.text(0.5, 0.5, "No measurements for this metric", ha="center", transform=ax.transAxes)
    env = report["environment"]
    config = report["configuration"]
    model = Path(config["model"]).name
    fig.suptitle(f"{model} | {env.get('gpu', 'CPU')} | {config.get('dtype', 'unspecified dtype')} | repeated-run medians",
                 y=0.99)
    if handles:
        fig.legend(handles.values(), [line.get_label() for line in handles.values()],
                   loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(6, len(handles)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        fig.savefig(file, format="png", dpi=160)
    plt.close(fig)
    return path
