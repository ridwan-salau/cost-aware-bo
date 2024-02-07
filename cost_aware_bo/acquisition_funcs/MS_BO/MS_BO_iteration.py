import copy

import torch
from botorch.acquisition import ExpectedImprovement

from cost_aware_bo.functions.iteration_funcs import get_gp_models
from cost_aware_bo.functions.processing_funcs import (
    get_gen_bounds,
    get_random_observations,
    normalize,
    standardize,
    unnormalize,
)
from cost_aware_bo.optimize_mem_acqf import optimize_acqf_by_mem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_initial_msbo_stages(new_x, n_init_stages, seed=None):
    newx_bounds = [[0] * n_init_stages, [1] * n_init_stages]
    X = get_random_observations(1, newx_bounds, seed)
    X = torch.cat((X, new_x), dim=1)
    return X


def get_msbo_bounds(bounds, last_stage_idx):
    b = [[], []]
    for idx in last_stage_idx:
        b[0].append(bounds[0][idx].item())
        b[1].append(bounds[1][idx].item())

    b = torch.tensor(b, device=DEVICE, dtype=torch.double)
    return b


def msbo_iteration(
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
    # GP is only trained on the last stage params

    last_stage_idx = params["h_ind"][-1]
    n_init_stage_idx = X.shape[1] - len(last_stage_idx)

    x = copy.deepcopy(X)

    x = x[:, last_stage_idx]
    b = get_msbo_bounds(bounds["x_cube"], last_stage_idx)

    train_x = normalize(x, bounds=b)
    train_y = standardize(y, bounds["y"])

    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)

    norm_bounds = get_gen_bounds(
        [last_stage_idx], params["normalization_bounds"], bound_type="norm"
    )

    acqf = ExpectedImprovement(model=gp_model, best_f=train_y.max())

    new_x, n_memoised, acq_value = optimize_acqf_by_mem(
        acqf=acqf,
        acqf_str=acqf_str,
        bounds=norm_bounds,
        iter=iter,
        params=params,
        seed=iter,
    )

    new_x = fill_initial_msbo_stages(new_x, n_init_stage_idx, iter)

    new_x = unnormalize(new_x, bounds=bounds["x_cube"])

    return new_x, n_memoised, acq_value
