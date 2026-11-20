"""Tamper-evident hash-chained JSONL audit log with model/data provenance."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GENESIS_HASH = "0" * 64


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_entry_hash(entry_without_hash: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(entry_without_hash).encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only SHA-256 hash-chained audit logger with provenance fields."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._prev_hash = GENESIS_HASH

        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            with self.log_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last = json.loads(lines[-1])
                self._seq = int(last["seq"]) + 1
                self._prev_hash = str(last["entry_hash"])

    def log_iteration(
        self,
        acq_type: str,
        seed: int,
        iteration: int,
        hparams: Dict[str, Any],
        stage0_cache_hit: bool,
        stage0_cache_key: str,
        stage_costs_sec: List[float],
        objective: float,
        eeipu_version: str = "",
        backbone_id: str = "",
        dataset_manifest_hash: str = "",
        model_weights_hash: str = "",
    ) -> Tuple[Dict[str, Any], float]:
        """Append one hash-chained audit record and return ``(record, write_ms)``."""
        t0 = time.perf_counter()
        record: Dict[str, Any] = {
            "seq": self._seq,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "eeipu_version": eeipu_version,
            "backbone_id": backbone_id,
            "dataset_manifest_hash": dataset_manifest_hash,
            "model_weights_hash": model_weights_hash,
            "acq_type": acq_type,
            "seed": seed,
            "iteration": iteration,
            "hparams": hparams,
            "stage0_cache_hit": stage0_cache_hit,
            "stage0_cache_key": stage0_cache_key,
            "stage_costs_sec": [round(float(c), 6) for c in stage_costs_sec],
            "objective": round(float(objective), 6),
            "prev_hash": self._prev_hash,
        }
        entry_hash = compute_entry_hash(record)
        record["entry_hash"] = entry_hash

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        write_ms = (time.perf_counter() - t0) * 1000.0
        self._seq += 1
        self._prev_hash = entry_hash
        return record, write_ms


def verify_audit_log(log_path: Path) -> Tuple[bool, Optional[str]]:
    """Verify hash-chain integrity of an audit log file.

    Returns:
        ``(True, None)`` if valid, otherwise ``(False, error_message)``.
    """
    path = Path(log_path)
    if not path.exists():
        return False, f"Audit log not found: {path}"

    expected_prev = GENESIS_HASH
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("seq") != idx:
                return False, f"Sequence mismatch at line {idx}: got {entry.get('seq')}"
            if entry.get("prev_hash") != expected_prev:
                return False, f"Chain break at seq {idx}: prev_hash mismatch"
            recorded_hash = entry.get("entry_hash")
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if compute_entry_hash(body) != recorded_hash:
                return False, f"Tampered entry at seq {idx}: entry_hash mismatch"
            expected_prev = recorded_hash

    return True, None


def verify_tamper_detection(log_path: Path) -> bool:
    """Copy the log, mutate one field in the middle entry, and confirm verification fails."""
    path = Path(log_path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return False

    mid = len(lines) // 2
    entry = json.loads(lines[mid])
    entry["objective"] = round(float(entry["objective"]) + 0.001, 6)
    lines[mid] = json.dumps(entry, sort_keys=True)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=True, encoding="utf-8") as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp.flush()
        ok, _ = verify_audit_log(Path(tmp.name))
        return not ok
