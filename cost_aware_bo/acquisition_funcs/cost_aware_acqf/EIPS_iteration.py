import torch
from botorch.acquisition.objective import IdentityMCObjective
from botorch.sampling import SobolQMCNormalSampler

from cost_aware_bo.functions.iteration_funcs import get_cost_model, get_gp_models
from cost_aware_bo.functions.processing_funcs import (
    get_cost_bounds,
    get_gen_bounds,
    normalize,
    standardize,
    unnormalize,
    unstandardize,
)
from cost_aware_bo.optimize_mem_acqf import optimize_acqf_by_mem

from .EIPS import EIPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eips_iteration(
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

    c = c.sum(axis=1).unsqueeze(-1)
    c = c.to(DEVICE)
    bounds = get_cost_bounds(c, bounds)
    cost_mll, cost_gp = get_cost_model(
        train_x, c, iter, params["h_ind"], bounds, acqf_str
    )

    cost_sampler = SobolQMCNormalSampler(sample_shape=params["cost_samples"], seed=iter)
    acqf = EIPS(
        acq_type=acqf_str,
        model=gp_model,
        cost_gp=cost_gp,
        best_f=train_y.max(),
        cost_sampler=cost_sampler,
        acq_objective=IdentityMCObjective(),
        unstandardizer=unstandardize,
        normalizer=normalize,
        unnormalizer=unnormalize,
        bounds=bounds,
        eta=decay,
        consumed_budget=consumed_budget,
        iter=iter,
        params=params,
    )

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
