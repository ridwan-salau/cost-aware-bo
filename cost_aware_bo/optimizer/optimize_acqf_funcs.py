"""Compatibility wrapper around BoTorch's ``optimize_acqf``."""

from botorch.optim import optimize_acqf as _botorch_optimize_acqf


def optimize_acqf(
    acq_function,
    bounds,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 512,
    acq_type: str = "",
    delta: int = 0,
    curr_iter: int = 0,
    **kwargs,
):
    """Thin compatibility wrapper delegating to ``botorch.optim.optimize_acqf``."""
    return _botorch_optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        **kwargs,
    )
