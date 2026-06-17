#!/usr/bin/env python3
"""Convert metrics summary CSV into paper-ready tables.

Examples:
  python scripts/export_paper_table.py logs/exports/metrics_csv/summary.csv
  python scripts/export_paper_table.py logs/exports/metrics_csv/summary.csv --output-dir logs/exports/paper_tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def infer_scenario(run_name: str) -> str:
    lower = run_name.lower()
    if "mppi" in lower:
        return "MPPI Corner Fail"
    if "dwb" in lower:
        return "DWB Oscillation"
    if "u_shape" in lower or "ushape" in lower:
        return "U-Shape Trap"
    if "door" in lower:
        return "Door Deadlock"
    return run_name.replace("_", " ").title()


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_rows(summary_csv: Path, method: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            run_name = raw["run_name"]
            rows.append(
                {
                    "Scenario": infer_scenario(run_name),
                    "Method": method,
                    "Run": run_name,
                    "Last Step": f"{parse_int(raw['last_step']):,}",
                    "Best Success (%)": f"{parse_float(raw['best_success']):.1f}",
                    "Final Success (%)": f"{parse_float(raw['final_success']):.1f}",
                    "Final Collision (%)": f"{parse_float(raw['final_collision']):.1f}",
                    "Final Timeout (%)": f"{parse_float(raw['final_timeout']):.1f}",
                    "Final Goal Dist. (m)": f"{parse_float(raw['final_avg_goal_end_dist']):.2f}",
                    "Final Return": f"{parse_float(raw['final_avg_return']):.2f}",
                    "Final Ep. Len": f"{parse_float(raw['final_avg_len']):.1f}",
                    "Phase": str(parse_int(raw["last_phase"])),
                    "Source Log": raw["source_log"],
                    "Detail CSV": raw["detail_csv"],
                }
            )
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_latex(rows: List[Dict[str, str]], output_path: Path, caption: str, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    column_spec = "l" * len(headers)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\centering\n")
        handle.write(f"\\caption{{{latex_escape(caption)}}}\n")
        handle.write(f"\\label{{{latex_escape(label)}}}\n")
        handle.write(f"\\begin{{tabular}}{{{column_spec}}}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(" & ".join(latex_escape(row[h]) for h in headers) + " \\\\\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table*}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert summary.csv into paper-ready tables")
    parser.add_argument("summary_csv", help="Path to summary.csv from export_metric_csv.py")
    parser.add_argument("--output-dir", default="logs/exports/paper_tables", help="Output directory")
    parser.add_argument("--method", default="DRL (PPO)", help="Method name shown in tables")
    parser.add_argument("--caption", default="Training results summary", help="LaTeX table caption")
    parser.add_argument("--label", default="tab:training-results", help="LaTeX table label")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    rows = build_rows(summary_csv, args.method)
    if not rows:
        raise SystemExit("No rows found in summary CSV")

    output_dir = Path(args.output_dir)
    stem = summary_csv.stem + "_paper"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_latex(rows, tex_path, args.caption, args.label)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
