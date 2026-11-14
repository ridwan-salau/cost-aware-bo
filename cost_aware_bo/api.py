"""High-level ask-and-tell API for memoization- and cost-aware Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .param_gen import generate_hps, update_dataset_new_run


@dataclass
class EEIPUOptimizer:
    """Ask-and-tell optimizer for multi-stage ML pipelines using EEIPU.

    Args:
        search_space: Mapping of ``"<stage_idx>__<param_name>"`` to
            ``{"lower": ..., "upper": ..., "dtype": "float" | "int"}``.
        n_init: Number of random warm-up evaluations before fitting GP surrogates.
        total_budget: Total cost budget across all iterations. If ``None``,
            cost cooling is disabled (equivalent to an unbounded budget, ``cost_cool=1``).
        acq_type: Acquisition function identifier (default ``"EEIPU"``).
            Supported: ``"EEIPU"``, ``"EI"``, ``"EIPS"``, ``"CArBO"``, ``"MS_CArBO"``.
        n_prefixes: Maximum number of top-performing prefixes to consider for memoization.
        epsilon: Small residual cost assigned to memoized (cached) stages.
        cost_samples: Number of Monte Carlo samples for expected inverse cost.
        seed: Random seed for reproducibility.
        device: Torch device string (``"cpu"`` or ``"cuda"``). Auto-detected if ``None``.
    """

    search_space: Dict[str, Dict[str, Any]]
    n_init: int = 5
    total_budget: Optional[float] = None
    acq_type: str = "EEIPU"
    n_prefixes: int = 5
    epsilon: float = 1e-2
    cost_samples: int = 500
    seed: int = 42
    device: Optional[str] = None

    dataset: Dict[str, Any] = field(default_factory=dict, init=False)
    iteration: int = field(default=0, init=False)
    consumed_budget: float = field(default=0.0, init=False)
    last_n_memoised: int = field(default=0, init=False)
    _x_bounds: Optional[List[List[Tuple[float, float]]]] = field(default=None, init=False)
    _budget_0: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.search_space:
            raise ValueError("search_space must be a non-empty dictionary.")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_params(self) -> Dict[str, Any]:
        if self.iteration >= self.n_init and self._budget_0 is None:
            self._budget_0 = self.consumed_budget

        effective_total_budget = (
            self.total_budget
            if self.total_budget is not None
            else max(self.consumed_budget * 2.0 + 1.0, 1e9)
        )
        return {
            "n_init_data": self.n_init,
            "total_budget": effective_total_budget,
            "budget_0": self._budget_0 or 0.0,
            "init_eta": 1.0,
            "decay_factor": 0.95,
            "normalization_bounds": [0, 1],
            "cost_samples": self.cost_samples,
            "epsilon": self.epsilon,
            "n_prefixes": self.n_prefixes,
            "use_pref_pool": 1 if self.acq_type == "EEIPU" else 0,
            "rand_seed": self.seed,
            "trial": 0,
        }

    def suggest(self) -> Dict[str, Union[float, int]]:
        """Suggest the next hyperparameter configuration to evaluate."""
        params = self._build_params()
        new_hp_dict, metadata = generate_hps(
            dataset=self.dataset,
            hp_sampling_range=self.search_space,
            iteration=self.iteration,
            params=params,
            exp_name="eeipu",
            trial=0,
            consumed_budget=self.consumed_budget,
            acq_type=self.acq_type,
        )
        self._x_bounds = metadata["x_bounds"]
        self.last_n_memoised = int(metadata.get("n_memoised", 0))
        return new_hp_dict

    def observe(
        self,
        hparams: Dict[str, Union[float, int]],
        objective: float,
        stage_costs: Sequence[float],
    ) -> None:
        """Record the observed objective (to maximize) and per-stage costs."""
        if self._x_bounds is None:
            # Allow direct observe() calls even before suggest() by deriving bounds
            stage_map: Dict[int, List[Tuple[float, float]]] = {}
            for key, spec in sorted(
                self.search_space.items(), key=lambda kv: int(kv[0].split("__")[0])
            ):
                stg = int(key.split("__")[0])
                stage_map.setdefault(stg, []).append((spec["lower"], spec["upper"]))
            self._x_bounds = [stage_map[k] for k in sorted(stage_map)]

        costs_list = [float(c) for c in stage_costs]
        self.consumed_budget += sum(costs_list)
        self.dataset = update_dataset_new_run(
            dataset=self.dataset,
            new_hp_dict=hparams,
            new_stg_costs=costs_list,
            new_obj=float(objective),
            x_bounds=self._x_bounds,
            acqf=self.acq_type,
            device=self.device,
        )
        self.iteration += 1

    def best(self) -> Tuple[Dict[str, float], float]:
        """Return ``(best_hparams, best_objective)`` observed so far."""
        if "y" not in self.dataset or len(self.dataset["y"]) == 0:
            raise RuntimeError("No observations recorded yet.")
        idx = int(torch.argmax(self.dataset["y"]).item())
        best_obj = float(self.dataset["y"][idx].item())
        best_hps = {
            name: float(vals[idx].item()) for name, vals in self.dataset["x"].items()
        }
        return best_hps, best_obj

