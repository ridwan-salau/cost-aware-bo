import copy
from itertools import chain

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf as botorch_optimize_acqf

from cost_aware_bo.optimizer.optimize_acqf_funcs import optimize_acqf as _compat_shim  # noqa: F401


class _MemoizedAcqWrapper(AcquisitionFunction):
    """Adapts (X, delta, curr_iter) acquisition functions to standard BoTorch forward(X)."""

    def __init__(self, base_acqf, delta: int, curr_iter: int) -> None:
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.delta = delta
        self.curr_iter = curr_iter

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.base_acqf(X, delta=self.delta, curr_iter=self.curr_iter)


def update_candidate(
    candidate, acqf_val, best_candidate, best_acqf_val, num_memoised, delta
):
    if acqf_val > best_acqf_val:
        return candidate.detach(), acqf_val, delta
    return best_candidate, best_acqf_val, num_memoised


def optimize_acqf_by_mem(
    acqf=None,
    acqf_str=None,
    bounds=None,
    iter=None,
    params=None,
    prefix_pool=[[]],
    seed=0,
):
    n_memoised = 0

    best_candidate, best_acqf_val = -torch.inf, -torch.inf
    for prefix in prefix_pool:
        cand_generation_bounds = copy.deepcopy(bounds)
        pref_stages = len(prefix)
        flat_prefix = list(chain(*prefix))
        fixed_features = (
            {i: float(val) for i, val in enumerate(flat_prefix)}
            if flat_prefix
            else None
        )

        wrapped_acqf = (
            _MemoizedAcqWrapper(acqf, delta=pref_stages, curr_iter=iter or 0)
            if acqf_str and "EIPU" in acqf_str
            else acqf
        )

        new_candidate, acqf_val = botorch_optimize_acqf(
            acq_function=wrapped_acqf,
            bounds=cand_generation_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            fixed_features=fixed_features,
            options={"seed": seed},
        )

        best_candidate, best_acqf_val, n_memoised = update_candidate(
            new_candidate,
            acqf_val.item(),
            best_candidate,
            best_acqf_val,
            n_memoised,
            pref_stages,
        )

    return best_candidate, n_memoised, best_acqf_val
