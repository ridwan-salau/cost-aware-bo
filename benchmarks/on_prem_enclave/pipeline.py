"""Clean-room 2-stage vision pipeline on public CIFAR-10 with prefix memoization."""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms


def select_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stage0_cache_key(backbone_id: str, stage0_hps: Dict[str, Any]) -> str:
    payload = {"backbone_id": backbone_id, **stage0_hps}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def hash_state_dict(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 fingerprint of a model's weights."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class TwoStageVisionPipeline:
    """Two-stage public CIFAR-10 pipeline with disk-backed Stage 0 memoization.

    Stage 0 (expensive, memoizable):
        Resizes public CIFAR-10 subset images to ``0__img_size`` (or 224 for ViT)
        and extracts frozen backbone embeddings for ``0__train_samples`` training
        images and a fixed validation slice.
    Stage 1 (cheaper, downstream):
        Trains a 2-layer MLP classifier on the extracted embeddings using
        ``1__lr``, ``1__weight_decay``, ``1__hidden_dim``, and ``1__epochs``.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        backbone_name: str = "resnet18",
        val_samples: int = 500,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backbone_name = backbone_name
        self.val_samples = val_samples
        self.device = select_device(device)
        self.seed = seed

        # Download/load public CIFAR-10 once outside timed stages
        self._train_raw = datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=True
        )
        self._val_raw = datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=True
        )

        # Deterministic dataset manifest fingerprint
        manifest = {
            "dataset": "torchvision.datasets.CIFAR10",
            "train_len": len(self._train_raw),
            "val_samples": self.val_samples,
            "val_targets_head": list(self._val_raw.targets[: self.val_samples]),
        }
        self.dataset_manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

        # Load frozen vision backbone once
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            backbone.fc = nn.Identity()
            self.backbone_id = f"torchvision.resnet18:{weights.name}"
        elif backbone_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT
            backbone = models.vit_b_16(weights=weights)
            backbone.heads = nn.Identity()
            self.backbone_id = f"torchvision.vit_b_16:{weights.name}"
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._backbone = backbone.to(self.device)

    @torch.no_grad()
    def _extract_features(
        self, raw_dataset: datasets.CIFAR10, n_samples: int, img_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_size = 224 if self.backbone_name == "vit_b_16" else img_size
        tfm = transforms.Compose(
            [
                transforms.Resize((effective_size, effective_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        indices = list(range(min(n_samples, len(raw_dataset))))
        subset = Subset(raw_dataset, indices)

        imgs = torch.stack([tfm(subset[i][0]) for i in range(len(subset))])
        labels = torch.tensor(
            [subset[i][1] for i in range(len(subset))], dtype=torch.long
        )

        feats_list = []
        batch_size = 64
        for start in range(0, len(imgs), batch_size):
            batch = imgs[start : start + batch_size].to(self.device)
            feats = self._backbone(batch).cpu()
            feats_list.append(feats)
        return torch.cat(feats_list, dim=0), labels

    def run_stage0(
        self, hparams: Dict[str, Any], enable_cache: bool = True
    ) -> Tuple[Dict[str, Any], float, float, bool, str]:
        stage0_hps = {k: v for k, v in hparams.items() if k.startswith("0__")}
        key = stage0_cache_key(self.backbone_id, stage0_hps)
        cache_file = self.cache_dir / f"stage0_{key}.pt"

        t0 = time.perf_counter()
        if enable_cache and cache_file.exists():
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            actual_cost = max(time.perf_counter() - t0, 1e-4)
            uncached_cost = float(payload.get("uncached_cost_sec", actual_cost))
            return payload, actual_cost, uncached_cost, True, key

        img_size = int(stage0_hps["0__img_size"])
        train_samples = int(stage0_hps["0__train_samples"])
        train_x, train_y = self._extract_features(
            self._train_raw, train_samples, img_size
        )
        val_x, val_y = self._extract_features(
            self._val_raw, self.val_samples, img_size
        )

        actual_cost = max(time.perf_counter() - t0, 1e-4)
        payload = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "uncached_cost_sec": actual_cost,
        }
        if enable_cache:
            torch.save(payload, cache_file)

        return payload, actual_cost, actual_cost, False, key

    def run_stage1(
        self, stage0_out: Dict[str, Any], hparams: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)

        lr = float(hparams["1__lr"])
        weight_decay = float(hparams["1__weight_decay"])
        hidden_dim = int(hparams["1__hidden_dim"])
        epochs = int(hparams["1__epochs"])

        in_dim = stage0_out["train_x"].shape[1]
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 10),
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        train_ds = TensorDataset(stage0_out["train_x"], stage0_out["train_y"])
        loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = stage0_out["val_x"].to(self.device)
            vy = stage0_out["val_y"].to(self.device)
            preds = model(vx).argmax(dim=1)
            val_acc = float((preds == vy).float().mean().item())

        weights_hash = hash_state_dict(model)
        cost_sec = max(time.perf_counter() - t0, 1e-4)
        return val_acc, cost_sec, weights_hash

    def evaluate(
        self, hparams: Dict[str, Any], enable_cache: bool = True
    ) -> Dict[str, Any]:
        stage0_out, c0_actual, c0_uncached, cache_hit, cache_key = self.run_stage0(
            hparams, enable_cache=enable_cache
        )
        val_acc, c1, weights_hash = self.run_stage1(stage0_out, hparams)
        return {
            "objective": val_acc,
            "stage_costs": [c0_actual, c1],
            "uncached_stage_costs": [c0_uncached, c1],
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "model_weights_hash": weights_hash,
            "backbone_id": self.backbone_id,
            "dataset_manifest_hash": self.dataset_manifest_hash,
        }
