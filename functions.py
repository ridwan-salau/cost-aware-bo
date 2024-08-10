import copy
import random
from collections import deque

import torch
from botorch.models import SingleTaskGP
from botorch.test_functions import (
    Ackley,
    Beale,
    Branin,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Rosenbrock,
    StyblinskiTang,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

SYNTHETIC_FUNCTIONS = {
    # Stage 1
    "branin2": Branin(negate=True, bounds=[[0, 10], [0, 10]]),
    "michale2": Michalewicz(negate=True),
    "styblnski2": StyblinskiTang(dim=2, negate=True),
    "beale2": Beale(negate=True),
    # Stage 2
    "ackley3": Ackley(dim=3, negate=True),
    "hartmann3": Hartmann(dim=3, negate=True),
    "styblnski3": StyblinskiTang(dim=3, negate=True),
    # Stage 3
    "rosenbrock2": Rosenbrock(negate=True),
    "levy2": Levy(negate=True),
    "holdertable2": HolderTable(negate=True),
}
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")


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


def logistic(x, params):  # logistic function
    return 1.0 / (1 + torch.exp(-x * params["slope"]))


def sin(x, params):  # sin
    return torch.sin(x)


def cos(x, params):  # cosine
    return torch.cos(x)


def poly(x, params):  # polynomial function
    return x ** params["degree"]


def apply(f, x, params={}, synthetic=False):
    if synthetic:
        return params["scale"] * f(x) + params["shift"]
    else:
        return params["scale"] * f(x, params) + params["shift"]


def cost2D(X, ctype=1):
    if ctype == 1:
        cost = apply(logistic, X[:, 0], {"slope": 5, "scale": 15, "shift": 20}) + apply(
            sin, X[:, 1], {"scale": 20, "shift": 25}
        )
    elif ctype == 2:
        cost = 5 * apply(cos, X[:, 0], {"scale": 10, "shift": 20}) - apply(
            sin, X[:, 1], {"scale": 25, "shift": 10}
        )

    elif ctype == 3:
        cost = apply(logistic, X[:, 0], {"slope": 5, "scale": 25, "shift": 10}) - apply(
            logistic, X[:, 1], {"slope": 3, "scale": 2, "shift": 3}
        )
    else:
        raise ValueError("Only cost types 1 to 3 acceptable")

    return cost.unsqueeze(-1)


def cost3D(X, ctype=1):
    if ctype == 1:
        cost = (
            apply(logistic, X[:, 0], {"scale": 5, "shift": 30, "slope": 30})
            - apply(sin, X[:, 1], {"scale": -12, "shift": 4})
            + apply(cos, X[:, 2], {"scale": 5, "shift": 30})
        )
    elif ctype == 2:
        cost = apply(sin, X[:, 1], {"scale": 15, "shift": 6}) + apply(
            logistic, X[:, 2], {"slope": 5, "scale": 27, "shift": 10}
        )
    elif ctype == 3:
        cost = (
            apply(logistic, X[:, 0], {"scale": 22, "shift": 5, "slope": 10})
            - apply(sin, X[:, 1], {"scale": -12, "shift": 5})
            + apply(cos, X[:, 2], {"scale": 22, "shift": 5})
        )
    else:
        raise ValueError("Only cost types 1 to 3 acceptable")

    return cost.unsqueeze(-1)


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


def get_dataset_bounds(X, Y, C, gen_bounds):
    bounds = {}
    bounds["x"] = gen_bounds + 0.0

    x_cube_bounds = [[], []]
    for i in range(X.shape[1]):
        x_cube_bounds[0].append(X[:, i].min().item())
        x_cube_bounds[1].append(X[:, i].max().item())
    bounds["x_cube"] = torch.tensor(x_cube_bounds, device=DEVICE)

    bounds["y"] = torch.tensor(
        [[Y.mean().item()], [Y.std().item()]], device=DEVICE, dtype=torch.double
    )

    std_c_bounds = [[], []]
    for stage_costs in C:
        log_sc = torch.log(stage_costs)
        std_c_bounds[0].append(log_sc.mean().item())
        std_c_bounds[1].append(log_sc.std().item())
    bounds["c"] = torch.tensor(std_c_bounds, device=DEVICE)

    return bounds


def get_random_observations(N=None, bounds=None):
    X = None
    # Generate initial training data, one dimension at a time
    for dim in range(len(bounds[0])):
        lo_bounds, hi_bounds = bounds[0][dim], bounds[1][dim]

        if torch.is_tensor(X):
            temp = torch.distributions.uniform.Uniform(lo_bounds, hi_bounds).sample(
                [N, 1]
            )
            X = torch.cat((X, temp), dim=1)
        else:
            X = torch.distributions.uniform.Uniform(lo_bounds, hi_bounds).sample([N, 1])
    return X


def generate_input_data(N=None, bounds=None, seed=0):
    torch.manual_seed(seed=seed)
    X = get_random_observations(N, bounds)
    return X


def F(X, params):
    funcs = params["obj_funcs"]
    param_idx = params["h_ind"]
    n_stages = len(param_idx)

    F = 0
    for stage in range(n_stages):
        f = funcs[stage]
        stage_params = param_idx[stage]

        obj = SYNTHETIC_FUNCTIONS[f]
        F += obj(X[:, stage_params])
    return F


def Cost_F(X, params):
    cost_types = params["cost_types"]
    param_idx = params["h_ind"]
    n_stages = len(param_idx)

    costs = []
    for stage in range(n_stages):
        ctype = cost_types[stage]
        stage_idx = param_idx[stage]

        if len(stage_idx) == 3:
            stage_cost = cost3D(X[:, stage_idx], ctype)
        elif len(stage_idx) == 2:
            stage_cost = cost2D(X[:, stage_idx], ctype)
        costs.append(stage_cost)
    return costs


def initialize_GP_model(X, y):
    X_, y_ = X + 0, y + 0
    gp_model = SingleTaskGP(X_, y_).to(X_)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    return mll, gp_model


def generate_prefix_pool(X, params):
    prefix_pool = []
    first_idx = params["n_init_data"]
    for i, param_config in enumerate(X[first_idx:]):
        prefix = []
        n_stages = len(params["h_ind"])
        for j in range(n_stages - 1):
            stage_params = params["h_ind"][j]
            prefix.append(list(param_config[stage_params].cpu().detach().numpy()))
            prefix_pool.append(copy.deepcopy(prefix))

    random.shuffle(prefix_pool)

    # Constant complexity to append at beginning of list
    prefix_pool = deque(prefix_pool)
    prefix_pool.appendleft([])
    prefix_pool = list(prefix_pool)

    if len(prefix_pool) > params["prefix_thresh"]:
        prefix_pool = prefix_pool[: params["prefix_thresh"]]

    return prefix_pool
