"""Reproducible clean-room benchmark comparing EEIPU, EI, EI+Cache, and Random Search."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from audit_log import AuditLogger, verify_audit_log, verify_tamper_detection
from eeipu import EEIPUOptimizer, __version__ as EEIPU_VERSION
from pipeline import TwoStageVisionPipeline

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "0__img_size": {"lower": 64, "upper": 160, "dtype": "int"},
    "0__train_samples": {"lower": 400, "upper": 1600, "dtype": "int"},
    "1__lr": {"lower": 1e-4, "upper": 1e-2, "dtype": "float"},
    "1__weight_decay": {"lower": 1e-5, "upper": 1e-2, "dtype": "float"},
    "1__hidden_dim": {"lower": 32, "upper": 256, "dtype": "int"},
    "1__epochs": {"lower": 5, "upper": 25, "dtype": "int"},
}


def sample_random_hparams(rng: random.Random) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, spec in SEARCH_SPACE.items():
        if spec["dtype"] == "int":
            out[key] = rng.randint(int(spec["lower"]), int(spec["upper"]))
        else:
            out[key] = rng.uniform(float(spec["lower"]), float(spec["upper"]))
    return out


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024.0 * 1024.0)


def run_single_trial(
    method: str,
    seed: int,
    budget_sec: float,
    n_init: int,
    pipeline: TwoStageVisionPipeline,
    output_dir: Path,
) -> Dict[str, Any]:
    # Isolate cache per trial so methods are compared fairly from a cold cache
    trial_cache = output_dir / "cache" / f"{method}_seed{seed}"
    if trial_cache.exists():
        shutil.rmtree(trial_cache)
    pipeline.cache_dir = trial_cache
    trial_cache.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "audit_logs" / f"{method}_seed{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    logger = AuditLogger(log_path)

    enable_cache = method in ("EEIPU", "EI_CACHE")
    bo_acq = "EI" if method == "EI_CACHE" else method

    rng = random.Random(seed)
    optimizer = (
        EEIPUOptimizer(
            search_space=SEARCH_SPACE,
            n_init=n_init,
            total_budget=budget_sec,
            acq_type=bo_acq,
            n_prefixes=5,
            epsilon=1e-2,
            cost_samples=256,
            seed=seed,
            device="cpu",  # Keep GP fitting on CPU for Apple Silicon / CPU parity
        )
        if bo_acq in ("EEIPU", "EI")
        else None
    )

    consumed = 0.0
    iteration = 0
    best_obj = float("-inf")
    cache_hits = 0
    uncached_c0_times: List[float] = []
    cached_c0_times: List[float] = []
    audit_write_ms_list: List[float] = []

    while consumed < budget_sec:
        if optimizer is None:
            hps = sample_random_hparams(rng)
        else:
            hps = optimizer.suggest()

        res = pipeline.evaluate(hps, enable_cache=enable_cache)
        obj = float(res["objective"])
        actual_costs = [float(c) for c in res["stage_costs"]]
        model_costs = [float(c) for c in res["uncached_stage_costs"]]

        consumed += sum(actual_costs)
        best_obj = max(best_obj, obj)
        if res["cache_hit"]:
            cache_hits += 1
            cached_c0_times.append(actual_costs[0])
        else:
            uncached_c0_times.append(actual_costs[0])

        if optimizer is not None:
            optimizer.observe(
                hps,
                objective=obj,
                stage_costs=model_costs,
                budget_cost=sum(actual_costs),
            )

        _, write_ms = logger.log_iteration(
            acq_type=method,
            seed=seed,
            iteration=iteration,
            hparams=hps,
            stage0_cache_hit=bool(res["cache_hit"]),
            stage0_cache_key=str(res["cache_key"]),
            stage_costs_sec=actual_costs,
            objective=obj,
            eeipu_version=EEIPU_VERSION,
            backbone_id=str(res["backbone_id"]),
            dataset_manifest_hash=str(res["dataset_manifest_hash"]),
            model_weights_hash=str(res["model_weights_hash"]),
        )
        audit_write_ms_list.append(write_ms)
        iteration += 1

    t_ver = time.perf_counter()
    ok, err = verify_audit_log(log_path)
    verify_ms = (time.perf_counter() - t_ver) * 1000.0
    if not ok:
        raise RuntimeError(f"Audit log verification failed: {err}")

    tamper_caught = verify_tamper_detection(log_path)
    if not tamper_caught:
        raise RuntimeError(f"Tamper detection test failed on {log_path}")

    return {
        "method": method,
        "seed": seed,
        "candidates_evaluated": iteration,
        "best_val_acc": best_obj,
        "consumed_sec": consumed,
        "stage0_cache_hits": cache_hits,
        "mean_stage0_uncached_sec": statistics.mean(uncached_c0_times)
        if uncached_c0_times
        else 0.0,
        "mean_stage0_cache_hit_sec": statistics.mean(cached_c0_times)
        if cached_c0_times
        else 0.0,
        "mean_audit_write_ms": statistics.mean(audit_write_ms_list)
        if audit_write_ms_list
        else 0.0,
        "audit_verify_ms": verify_ms,
        "cache_disk_mb": dir_size_mb(trial_cache),
        "audit_verified": ok,
        "tamper_detection_passed": tamper_caught,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean-room on-premises benchmark for EEIPU vs EI vs EI+Cache vs RAND."
    )
    parser.add_argument("--budget-sec", type=float, default=120.0)
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["RAND", "EI", "EI_CACHE", "EEIPU"],
        choices=["RAND", "EI", "EI_CACHE", "EEIPU"],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "vit_b_16"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TwoStageVisionPipeline(
        data_dir=args.data_dir,
        cache_dir=args.output_dir / "cache",
        backbone_name=args.backbone,
        device=args.device,
    )

    all_trials: List[Dict[str, Any]] = []
    for method in args.methods:
        for seed in args.seeds:
            print(
                f"Running method={method} seed={seed} budget={args.budget_sec}s "
                f"n_init={args.n_init} backbone={args.backbone}..."
            )
            trial = run_single_trial(
                method=method,
                seed=seed,
                budget_sec=args.budget_sec,
                n_init=args.n_init,
                pipeline=pipeline,
                output_dir=args.output_dir,
            )
            print("  ->", trial)
            all_trials.append(trial)

    summary: Dict[str, Dict[str, Any]] = {}
    for method in args.methods:
        rows = [t for t in all_trials if t["method"] == method]
        cands = [r["candidates_evaluated"] for r in rows]
        accs = [r["best_val_acc"] for r in rows]
        hits = [r["stage0_cache_hits"] for r in rows]
        uncached_s = [r["mean_stage0_uncached_sec"] for r in rows if r["mean_stage0_uncached_sec"] > 0]
        cached_s = [r["mean_stage0_cache_hit_sec"] for r in rows if r["mean_stage0_cache_hit_sec"] > 0]
        write_ms = [r["mean_audit_write_ms"] for r in rows]
        verify_ms = [r["audit_verify_ms"] for r in rows]
        disk_mb = [r["cache_disk_mb"] for r in rows]

        summary[method] = {
            "mean_candidates": statistics.mean(cands),
            "std_candidates": statistics.stdev(cands) if len(cands) > 1 else 0.0,
            "mean_best_acc": statistics.mean(accs),
            "std_best_acc": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "mean_cache_hits": statistics.mean(hits),
            "mean_stage0_uncached_sec": statistics.mean(uncached_s) if uncached_s else 0.0,
            "mean_stage0_cache_hit_sec": statistics.mean(cached_s) if cached_s else 0.0,
            "mean_audit_write_ms": statistics.mean(write_ms),
            "mean_audit_verify_ms": statistics.mean(verify_ms),
            "mean_cache_disk_mb": statistics.mean(disk_mb),
            "all_audits_verified": all(r["audit_verified"] for r in rows),
            "all_tamper_tests_passed": all(r["tamper_detection_passed"] for r in rows),
        }

    with (args.output_dir / "trials.json").open("w", encoding="utf-8") as f:
        json.dump(all_trials, f, indent=2)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + list(next(iter(summary.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, stats in summary.items():
            writer.writerow({"method": method, **stats})

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
