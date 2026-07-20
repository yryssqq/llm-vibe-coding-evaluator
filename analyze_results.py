#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from llm_vibe_eval import analytics_engine  # noqa: E402


def load_results_csv(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["passed"] = df["passed"].astype(bool)
    df["unsafe"] = df["unsafe_patterns"].fillna("").astype(bool)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--output-dir", default="results/report")
    args = parser.parse_args()

    df = load_results_csv(args.csv_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_evaluations": int(len(df)),
        "overall_pass_at_1": analytics_engine.pass_at_1(df),
        "pass_at_1_by_model": analytics_engine.pass_at_1(df, ["model"]).to_dict(orient="records"),
        "pass_at_1_by_complexity": analytics_engine.pass_at_1(df, ["complexity"]).to_dict(orient="records"),
        "error_distribution": analytics_engine.error_distribution(df).to_dict(orient="records"),
        "unsafe_code_rate_overall": analytics_engine.unsafe_code_rate(df),
        "unsafe_code_rate_by_model": analytics_engine.unsafe_code_rate(df, ["model"]).to_dict(orient="records"),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    heatmap_path = analytics_engine.plot_error_complexity_heatmap(df, output_dir / "error_vs_complexity.png")
    barplot_path = analytics_engine.plot_pass_rate_by_model(df, output_dir / "pass_rate_by_model.png")

    print(f"\nSummary written to: {summary_path}")
    print(f"Heatmap written to: {heatmap_path}")
    print(f"Bar chart written to: {barplot_path}")


if __name__ == "__main__":
    main()
