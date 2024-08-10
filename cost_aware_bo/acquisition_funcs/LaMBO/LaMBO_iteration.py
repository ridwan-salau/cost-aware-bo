from botorch.acquisition import UpperConfidenceBound

from cost_aware_bo.functions.iteration_funcs import get_gp_models
from cost_aware_bo.functions.processing_funcs import (
    get_gen_bounds,
    normalize,
    standardize,
    unnormalize,
)
from cost_aware_bo.optimize_mem_acqf import optimize_acqf_by_mem


def lambo_iteration(
    X,
    y,
    c,
    bounds=None,
    acqf_str="",
    decay=None,
    iter=None,
    consumed_budget=None,
    params=None,
):
    train_x = normalize(X, bounds=bounds["x_cube"])
    train_y = standardize(y, bounds["y"])

    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)

    norm_bounds = get_gen_bounds(
        params["h_ind"], params["normalization_bounds"], bound_type="norm"
    )

    acqf = UpperConfidenceBound(model=gp_model, beta=0.2, maximize=True)

    new_x, n_memoised, acq_value = optimize_acqf_by_mem(
        acqf=acqf,
        acqf_str=acqf_str,
        bounds=norm_bounds,
        iter=iter,
        params=params,
        seed=iter,
    )

    new_x = unnormalize(new_x, bounds=bounds["x_cube"])

    return new_x, n_memoised, acq_value
