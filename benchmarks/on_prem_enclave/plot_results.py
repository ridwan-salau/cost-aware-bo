"""Generate publication-grade PGFPlots figure (and optional PNG) from JSONL audit logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LABELS = {
    "RAND": "Random Search (RAND)",
    "EI": "Expected Improvement (EI)",
    "EI_CACHE": "EI + Stage 0 Cache",
    "EEIPU": "EEIPU (Memoization-Aware)",
}

PGF_STYLES = {
    "RAND": "color=gray!80!black, dashed, line width=1.1pt",
    "EI": "color=blue!70!black, dashdotted, line width=1.1pt",
    "EI_CACHE": "color=orange!85!black, dotted, line width=1.3pt",
    "EEIPU": "color=teal!80!black, solid, line width=1.6pt",
}


def load_trajectory(log_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cum_time_sec, best_acc_so_far, cand_count)`` from one JSONL log."""
    times: List[float] = []
    best_accs: List[float] = []
    counts: List[float] = []

    cum_t = 0.0
    best_a = 0.0
    with log_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cum_t += sum(float(c) for c in rec["stage_costs_sec"])
            best_a = max(best_a, float(rec["objective"]))
            times.append(cum_t)
            best_accs.append(best_a)
            counts.append(float(idx))

    return np.array(times), np.array(best_accs), np.array(counts)


def step_interpolate(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation onto a uniform time grid."""
    out = np.full_like(grid, np.nan, dtype=float)
    for i, t in enumerate(grid):
        idx = np.searchsorted(times, t, side="right") - 1
        if idx >= 0:
            out[i] = values[idx]
    return out


def _coords_str(xs: np.ndarray, ys: np.ndarray) -> str:
    pts = [f"({x:.2f},{y:.4f})" for x, y in zip(xs, ys) if not np.isnan(y)]
    return " ".join(pts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PGFPlots LaTeX snippet from verified audit logs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--budget-sec", type=float, default=120.0)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["RAND", "EI", "EI_CACHE", "EEIPU"],
    )
    args = parser.parse_args()

    audit_dir = args.results_dir / "audit_logs"
    grid = np.linspace(0.0, args.budget_sec, 61)

    acc_plots: List[str] = []
    cnt_plots: List[str] = []
    min_acc = 1.0
    max_acc = 0.0
    max_cnt = 1.0

    for method in args.methods:
        log_files = sorted(audit_dir.glob(f"{method}_seed*.jsonl"))
        if not log_files:
            continue

        acc_curves = []
        cnt_curves = []
        for lf in log_files:
            t, a, c = load_trajectory(lf)
            acc_curves.append(step_interpolate(t, a, grid))
            cnt_curves.append(step_interpolate(t, c, grid))

        acc_arr = np.vstack(acc_curves)
        cnt_arr = np.nan_to_num(np.vstack(cnt_curves), nan=0.0)

        acc_mean = np.nanmean(acc_arr, axis=0) * 100.0  # percent
        cnt_mean = np.mean(cnt_arr, axis=0)

        valid_acc = acc_mean[~np.isnan(acc_mean)]
        if len(valid_acc):
            min_acc = min(min_acc, float(np.min(valid_acc)))
            max_acc = max(max_acc, float(np.max(valid_acc)))
        max_cnt = max(max_cnt, float(np.max(cnt_mean)))

        style = PGF_STYLES.get(method, "color=black")
        label = LABELS.get(method, method)
        acc_plots.append(
            f"\\addplot[{style}] coordinates {{{_coords_str(grid, acc_mean)}}};\n"
            f"\\addlegendentry{{{label}}}"
        )
        cnt_plots.append(
            f"\\addplot[{style}] coordinates {{{_coords_str(grid, cnt_mean)}}};\n"
            f"\\addlegendentry{{{label}}}"
        )

    ymin_acc = max(0.0, np.floor(min_acc - 1.5))
    ymax_acc = min(100.0, np.ceil(max_acc + 1.5))
    ymax_cnt = int(np.ceil(max_cnt * 1.12))

    tex = rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  name=leftplot,
  width=0.47\textwidth,
  height=5.2cm,
  title={{(a) Best Validation Accuracy vs.\ Budget}},
  xlabel={{Elapsed Compute Budget (s)}},
  ylabel={{Best Val.\ Accuracy (\%)}},
  xmin=0, xmax={args.budget_sec:.0f},
  ymin={ymin_acc:.0f}, ymax={ymax_acc:.0f},
  grid=both,
  grid style={{dashed, gray!25}},
  legend pos=south east,
  legend style={{font=\scriptsize, fill=white, fill opacity=0.85}},
  tick label style={{font=\footnotesize}},
  label style={{font=\small}},
  title style={{font=\small\bfseries}}
]
{chr(10).join(acc_plots)}
\end{{axis}}
\begin{{axis}}[
  at={{(leftplot.outer east)}},
  anchor=outer west,
  xshift=0.4cm,
  width=0.47\textwidth,
  height=5.2cm,
  title={{(b) Candidates Evaluated vs.\ Budget}},
  xlabel={{Elapsed Compute Budget (s)}},
  ylabel={{Cumulative Candidates}},
  xmin=0, xmax={args.budget_sec:.0f},
  ymin=0, ymax={ymax_cnt},
  grid=both,
  grid style={{dashed, gray!25}},
  legend pos=north west,
  legend style={{font=\scriptsize, fill=white, fill opacity=0.85}},
  tick label style={{font=\footnotesize}},
  label style={{font=\small}},
  title style={{font=\small\bfseries}}
]
{chr(10).join(cnt_plots)}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Mean validation accuracy and cumulative candidate throughput across seeds versus elapsed compute budget on public CIFAR-10 (Apple M1).}}
\label{{fig:benchmark}}
\end{{figure}}
"""
    out_tex = args.results_dir / "fig_benchmark.tex"
    out_tex.write_text(tex, encoding="utf-8")
    print(f"Saved {out_tex}")


if __name__ == "__main__":
    main()
