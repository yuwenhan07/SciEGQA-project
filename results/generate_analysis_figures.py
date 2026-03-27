from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "static" / "images"

COLORS = {
    "whole": "#cc5c4c",
    "page": "#0f7b7b",
    "crop": "#ef8c3a",
    "accent": "#0a5155",
    "muted": "#7a9258",
    "grid": "#d8dcd5",
    "panel": "#f7f7f2",
}

NAME_FIXES = {
    "Qwen35-122B-A10B": "Qwen3.5-122B-A10B",
    "Erine-5": "Ernie-5",
    "Claude Sonnent 4.6": "Claude Sonnet 4.6",
}


def normalize_model(name: str) -> str:
    return NAME_FIXES.get(name.strip(), name.strip())


def parse_pct(text: str) -> float:
    return float(text.strip().rstrip("%"))


def load_result_lines(preferred_names: list[str]) -> tuple[list[str], str]:
    for name in preferred_names:
        path = RESULTS_DIR / name
        if path.exists():
            delimiter = "," if path.suffix == ".csv" else "\t"
            return path.read_text().splitlines(), delimiter
    raise FileNotFoundError(f"Could not find any of: {', '.join(preferred_names)}")


def load_granularity_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    lines, delimiter = load_result_lines(["task2-results.csv", "task2.md"])
    rows = []
    breakdown_rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split(delimiter)
        model = normalize_model(parts[0])
        rows.append(
            {
                "model": model,
                "whole": parse_pct(parts[8]),
                "page": parse_pct(parts[16]),
                "crop": parse_pct(parts[24]),
            }
        )
        breakdown_rows.extend(
            [
                {"model": model, "granularity": "Whole document", "reasoning": "SPSR", "accuracy": parse_pct(parts[2])},
                {"model": model, "granularity": "Whole document", "reasoning": "SPMR", "accuracy": parse_pct(parts[4])},
                {"model": model, "granularity": "Whole document", "reasoning": "MPMR", "accuracy": parse_pct(parts[6])},
                {"model": model, "granularity": "Evidence page(s)", "reasoning": "SPSR", "accuracy": parse_pct(parts[10])},
                {"model": model, "granularity": "Evidence page(s)", "reasoning": "SPMR", "accuracy": parse_pct(parts[12])},
                {"model": model, "granularity": "Evidence page(s)", "reasoning": "MPMR", "accuracy": parse_pct(parts[14])},
                {"model": model, "granularity": "Evidence crop(s)", "reasoning": "SPSR", "accuracy": parse_pct(parts[18])},
                {"model": model, "granularity": "Evidence crop(s)", "reasoning": "SPMR", "accuracy": parse_pct(parts[20])},
                {"model": model, "granularity": "Evidence crop(s)", "reasoning": "MPMR", "accuracy": parse_pct(parts[22])},
            ]
        )
    return pd.DataFrame(rows), pd.DataFrame(breakdown_rows)


def load_grounding_results() -> pd.DataFrame:
    lines, delimiter = load_result_lines(["task1-results.csv", "task1.md"])
    rows = []
    start_index = 1 if delimiter == "," else 2
    for line in lines[start_index:]:
        if not line.strip():
            continue
        parts = line.split(delimiter)
        rows.append(
            {
                "model": normalize_model(parts[0]),
                "valid_output": int(parts[1]),
                "valid_ratio": parse_pct(parts[2]),
                "mean_iou": parse_pct(parts[3] if delimiter == "," else parts[6]),
                "iou03": parse_pct(parts[4] if delimiter == "," else parts[7]),
                "iou05": parse_pct(parts[5] if delimiter == "," else parts[8]),
                "iou07": parse_pct(parts[6] if delimiter == "," else parts[9]),
                "acc": parse_pct(parts[7] if delimiter == "," else parts[10]),
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bfc7c1")
    ax.spines["bottom"].set_color("#bfc7c1")


def plot_task2_totals(granularity_results: pd.DataFrame) -> None:
    df = granularity_results.sort_values("crop")
    y = np.arange(len(df))
    height = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=200)
    style_axes(ax)
    ax.barh(y - height, df["whole"], height=height, color=COLORS["whole"], label="Whole document")
    ax.barh(y, df["page"], height=height, color=COLORS["page"], label="Evidence page(s)")
    ax.barh(y + height, df["crop"], height=height, color=COLORS["crop"], label="Evidence crop(s)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlim(0, 90)
    ax.set_xlabel("Answer accuracy (%)")
    ax.set_title("Task 2: QA Accuracy under different input granularities", fontsize=16, loc="left", pad=14)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results-task2-granularity.png", bbox_inches="tight")
    plt.close(fig)


def plot_task2_breakdown(granularity_breakdown: pd.DataFrame) -> None:
    average_pivot = (
        granularity_breakdown.groupby(["reasoning", "granularity"], as_index=False)["accuracy"]
        .mean()
        .pivot(index="reasoning", columns="granularity", values="accuracy")
        .loc[["SPSR", "SPMR", "MPMR"], ["Whole document", "Evidence page(s)", "Evidence crop(s)"]]
    )

    difficulty_df = (
        granularity_breakdown.pivot_table(index=["model", "reasoning"], columns="granularity", values="accuracy")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    difficulty_df["gain"] = difficulty_df["Evidence crop(s)"] - difficulty_df["Whole document"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.6, 9.8),
        dpi=200,
        gridspec_kw={"height_ratios": [1, 1.55]},
        constrained_layout=True,
    )

    ax = axes[0]
    heatmap = ax.imshow(average_pivot.values, cmap="YlGnBu", vmin=0, vmax=90, aspect="auto")
    ax.set_title("Task 2: Average accuracy by reasoning type and input granularity", fontsize=16, loc="left", pad=12)
    ax.set_xticks(range(len(average_pivot.columns)))
    ax.set_xticklabels(average_pivot.columns, rotation=12, ha="right", fontsize=13)
    ax.set_yticks(range(len(average_pivot.index)))
    ax.set_yticklabels(average_pivot.index, fontsize=13)
    for i in range(average_pivot.shape[0]):
        for j in range(average_pivot.shape[1]):
            value = average_pivot.iloc[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="white" if value > 55 else COLORS["accent"], fontsize=10.8, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    style_axes(ax)
    type_styles = {
        "SPSR": {"color": COLORS["page"], "marker": "o"},
        "SPMR": {"color": COLORS["crop"], "marker": "s"},
        "MPMR": {"color": COLORS["whole"], "marker": "^"},
    }
    for reasoning, style in type_styles.items():
        subset = difficulty_df[difficulty_df["reasoning"] == reasoning]
        sizes = 85 + subset["gain"] * 4
        ax.scatter(
            subset["Whole document"],
            subset["Evidence crop(s)"],
            s=sizes,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.9,
            label=reasoning,
        )
        center_x = subset["Whole document"].mean()
        center_y = subset["Evidence crop(s)"].mean()
        ax.scatter(
            center_x,
            center_y,
            s=300,
            color=style["color"],
            marker="*",
            edgecolor=COLORS["accent"],
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            reasoning,
            (center_x, center_y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10,
            color=COLORS["accent"],
            fontweight="bold",
        )

    ax.plot([0, 90], [0, 90], linestyle="--", linewidth=1.1, color="#8ca5a8", alpha=0.9)
    ax.set_xlim(0, 90)
    ax.set_ylim(48, 90)
    ax.set_xlabel("Whole-document accuracy (%)")
    ax.set_ylabel("Cropped-evidence accuracy (%)")
    ax.set_title("Task 2: Question types form distinct difficulty bands", fontsize=16, loc="left", pad=12)
    ax.text(
        0.02,
        0.96,
        "Lower and closer left indicates harder question types.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["accent"],
    )
    ax.legend(frameon=False, ncol=3, loc="lower right")

    cbar = fig.colorbar(heatmap, ax=axes[0], fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(OUTPUT_DIR / "results-task2-breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_task1_relationship(grounding_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=200)

    ax = axes[0]
    style_axes(ax)
    sizes = 60 + (grounding_results["valid_ratio"] - grounding_results["valid_ratio"].min()) * 12
    scatter = ax.scatter(
        grounding_results["mean_iou"],
        grounding_results["acc"],
        s=sizes,
        c=grounding_results["valid_ratio"],
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )
    corr = np.corrcoef(grounding_results["mean_iou"], grounding_results["acc"])[0, 1]
    ax.set_title("Task 1: Better grounding strongly tracks better final answers", fontsize=14, loc="left", pad=12)
    ax.set_xlabel("Mean IoU (%)")
    ax.set_ylabel("Final answer accuracy (%)")
    ax.text(0.02, 0.96, f"Pearson r = {corr:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=COLORS["accent"])

    label_positions = [
        (22.5, 53.0),
        (22.5, 50.5),
        (22.5, 48.0),
    ]
    for (text_x, text_y), (_, row) in zip(label_positions, grounding_results.nlargest(3, "acc").iterrows()):
        ax.annotate(
            row["model"],
            (row["mean_iou"], row["acc"]),
            xytext=(text_x, text_y),
            textcoords="data",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["accent"],
                "lw": 1.0,
                "shrinkA": 0,
                "shrinkB": 6,
            },
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            fontsize=8.8,
            color=COLORS["accent"],
            ha="left",
            va="center",
        )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Valid output ratio (%)")

    ax = axes[1]
    threshold_df = grounding_results.sort_values("iou03", ascending=False).head(8)
    y = np.arange(len(threshold_df))
    height = 0.22
    style_axes(ax)
    ax.barh(y - height, threshold_df["iou03"], height=height, color="#98c77a", label="IoU@0.3")
    ax.barh(y, threshold_df["iou05"], height=height, color=COLORS["page"], label="IoU@0.5")
    ax.barh(y + height, threshold_df["iou07"], height=height, color=COLORS["accent"], label="IoU@0.7")
    ax.set_yticks(y)
    ax.set_yticklabels(threshold_df["model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Threshold hit rate (%)")
    ax.set_title("High-IoU localization remains the main bottleneck", fontsize=13.5, loc="left", pad=14)
    ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=9.5, borderaxespad=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results-task1-grounding.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    granularity_results, granularity_breakdown = load_granularity_results()
    grounding_results = load_grounding_results()
    plot_task2_totals(granularity_results)
    plot_task2_breakdown(granularity_breakdown)
    plot_task1_relationship(grounding_results)
    print("Generated analysis figures in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
