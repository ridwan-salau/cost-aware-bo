# On-Premises / Enclave Clean-Room Benchmark

This benchmark evaluates `eeipu` (`EEIPUOptimizer`) on a 2-stage vision pipeline using **public CIFAR-10 data only**, running completely offline on local hardware (Mac CPU/MPS or Linux CPU/GPU) with:

- **Stage 0 (memoizable):** Frozen ResNet-18 (or `vit_b_16`) feature extraction with disk-backed content-addressed caching (`sha256` of backbone ID + Stage 0 hyperparameters).
- **Stage 1 (downstream):** 2-layer MLP classifier training and validation accuracy evaluation.
- **Baselines & Ablation:** `RAND`, `EI` (`LogExpectedImprovement`), `EI_CACHE` (passive Stage 0 disk cache ablation), and `EEIPU`.
- **Tamper-evident provenance audit log:** Append-only SHA-256 hash-chained JSONL (`results/audit_logs/*.jsonl`) recording `eeipu_version`, `backbone_id`, `dataset_manifest_hash`, `model_weights_hash`, hyperparameters, cache status, stage costs, and validation accuracy, verified after every trial with `verify_audit_log` and `verify_tamper_detection`.

## Reproduce Results & Technical Note

```bash
pip install -e ../..
pip install torchvision
python run_benchmark.py --budget-sec 120 --n-init 6 --seeds 0 1 2
python plot_results.py --results-dir ./results --budget-sec 120
pdflatex technical_note.tex && pdflatex technical_note.tex
```

Outputs are written to `results/summary.json`, `results/summary.csv`, `results/trials.json`, `results/fig_benchmark.tex`, `results/audit_logs/`, and `technical_note.pdf`.
