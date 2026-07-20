from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .models import EvaluationResult

sns.set_theme(style="whitegrid")


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": r.task_id,
                "model": r.model,
                "complexity": r.complexity.value,
                "passed": r.passed,
                "error_type": r.error_type.value,
                "error_message": r.error_message,
                "duration_seconds": r.duration_seconds,
                "unsafe": len(r.unsafe_patterns) > 0,
            }
            for r in results
        ]
    )


def pass_at_1(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame | float:
    if group_by:
        return df.groupby(group_by)["passed"].mean().rename("pass@1").reset_index()
    return float(df["passed"].mean())


def error_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[~df["passed"]]["error_type"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .rename(columns={"index": "error_type"})
    )


def unsafe_code_rate(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame | float:
    if group_by:
        return df.groupby(group_by)["unsafe"].mean().rename("unsafe_rate").reset_index()
    return float(df["unsafe"].mean())


def build_summary_report(df: pd.DataFrame) -> dict:
    return {
        "n_evaluations": int(len(df)),
        "overall_pass_at_1": pass_at_1(df),
        "pass_at_1_by_model": pass_at_1(df, ["model"]).to_dict(orient="records"),
        "pass_at_1_by_complexity": pass_at_1(df, ["complexity"]).to_dict(orient="records"),
        "error_distribution": error_distribution(df).to_dict(orient="records"),
        "unsafe_code_rate_overall": unsafe_code_rate(df),
        "unsafe_code_rate_by_model": unsafe_code_rate(df, ["model"]).to_dict(orient="records"),
    }


def plot_error_complexity_heatmap(df: pd.DataFrame, output_path: str | Path) -> Path:
    failures = df[~df["passed"]]
    complexity_order = [c for c in ["easy", "medium", "hard"] if c in df["complexity"].unique()]
    if failures.empty:
        pivot = pd.DataFrame(0, index=["none"], columns=complexity_order)
    else:
        pivot = pd.crosstab(failures["error_type"], failures["complexity"])
        pivot = pivot.reindex(columns=complexity_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="rocket_r", ax=ax, cbar_kws={"label": "count"}, vmin=0)
    ax.set_title("Error Type vs. Task Complexity")
    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Error Type")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_pass_rate_by_model(df: pd.DataFrame, output_path: str | Path) -> Path:
    rates = pass_at_1(df, ["model"]).sort_values("pass@1", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rates, x="pass@1", y="model", hue="model", legend=False, palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("pass@1 by Model")
    ax.set_xlabel("pass@1")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
