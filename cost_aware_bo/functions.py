import copy
import random
from collections import deque
from typing import Dict, List

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(data, bounds=None):
    data_ = data + 0
    dims = data_.shape[1]

    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data_[:, dim] = (data_[:, dim] - mn) / (mx - mn)
    return data_


def unnormalize(data, bounds=None):
    data_ = data + 0
    dims = data_.shape[1]
    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data_[:, dim] = (data_[:, dim] * (mx - mn)) + mn
    return data_


def normalize_cost(data, params):
    mn, mx = 10, 300
    data = (data - mn) / (mx - mn)
    alpha, eps = params["alpha"], params["norm_eps"]
    data = alpha * data + eps
    try:
        assert data.min() > 0
    except AssertionError:
        print(
            f"EXCEPTION RAISED BECAUSE THE MINIMUM DATAPOINT IS = {data.min().item()}, MAXIMUM FOR SOME REASON IS = {data.max().item()}, SHAPE IS = {data.shape}, AND NUMBER OF NANS IS {torch.isnan(data.view(-1)).sum().item()}"
        )
    return data


def standardize(data, bounds=None):
    data_ = data + 0
    mean, std = bounds[0].item(), bounds[1].item()
    data_ = (data_ - mean) / std
    return data_


def unstandardize(data, bounds=None):
    data_ = data + 0
    mean, std = bounds[0].item(), bounds[1].item()
    data_ = (data_ * std) + mean
    return data_


def get_gen_bounds(param_idx, func_bounds, funcs=None, bound_type=""):
    lo_bounds, hi_bounds = [], []

    for stage in range(len(param_idx)):
        if bound_type == "norm":
            f_bounds = func_bounds
        else:
            f = funcs[stage]
            f_bounds = func_bounds[f]
        stage_size = len(param_idx[stage])

        lo_bounds += [f_bounds[0]] * stage_size
        hi_bounds += [f_bounds[1]] * stage_size

    bounds = torch.tensor([lo_bounds, hi_bounds], device=DEVICE, dtype=torch.double)
    return bounds


def get_dataset_bounds(X: Dict[str, List], Y, C, gen_bounds):
    bounds = {}
    bounds["x"] = gen_bounds + 0.0

    x_cube_bounds = [
        [min(hp_values) for hp_values in X.values()],
        [max(hp_values) for hp_values in X.values()],
    ]
    # for i in range(X.shape[1]):
    #     x_cube_bounds[0].append(X[:,i].min().item())
    #     x_cube_bounds[1].append(X[:,i].max().item())
    bounds["x_cube"] = torch.tensor(x_cube_bounds, device=DEVICE)

    bounds["y"] = torch.tensor(
        [[Y.mean().item()], [Y.std().item()]], device=DEVICE, dtype=torch.double
    )

    std_c_bounds = [[], []]
    for stage in range(C.shape[1]):
        stage_costs = C[:, stage]
        # print("Stage costs:", stage_costs)
        log_sc = torch.log(stage_costs)
        std_c_bounds[0].append(log_sc.mean().item())
        std_c_bounds[1].append(log_sc.std().item())
    bounds["c"] = torch.tensor(std_c_bounds, device=DEVICE)

    c_cube = [[], []]
    for stage in range(C.shape[1]):
        stage_costs = C[:, stage]
        c_cube[0].append(stage_costs.min().item())
        c_cube[1].append(stage_costs.max().item())
    bounds["c_cube"] = torch.tensor(c_cube, device=DEVICE)

    return bounds


def initialize_GP_model(X, y, params=None):
    X_, y_ = X + 0, y + 0
    gp_model = SingleTaskGP(X_, y_).to(X_)
    gp_model = gp_model.to(DEVICE)
    # if params is not None:
    #     kernel = params['kernel']
    #     kernel = KERNELS[kernel]
    #     gp_model.covar_module = ScaleKernel(kernel).to(DEVICE)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(DEVICE)
    return mll, gp_model


def generate_prefix_pool(X, acqf, params):
    prefix_pool = []
    first_idx = params["n_init_data"]

    if acqf != "EEIPU":
        prefix_pool.append([])
        return prefix_pool

    for i, param_config in enumerate(X[first_idx:]):
        prefix = []
        n_stages = len(params["h_ind"])
        for j in range(n_stages - 1):
            stage_params = params["h_ind"][j]
            # print(j, "Stage params:", stage_params, param_config[stage_params])
            prefix.append(list(param_config[stage_params].cpu().detach().numpy()))
            prefix_pool.append(copy.deepcopy(prefix))
            # print(j, "Prefix: ", prefix)

    random.shuffle(prefix_pool)

    # Constant complexity to append at beginning of list
    prefix_pool = deque(prefix_pool)
    prefix_pool.appendleft([])
    prefix_pool = list(prefix_pool)

    if len(prefix_pool) > params["prefix_thresh"]:
        prefix_pool = prefix_pool[: params["prefix_thresh"]]

    return prefix_pool
