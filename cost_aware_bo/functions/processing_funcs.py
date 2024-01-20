import torch
import copy
from typing import Dict, List

# from functions.synthetic_functions import Cost_F, F # We shouldn't have this here, I think
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MS_ACQFS = ["EEIPU", "MS_CArBO", "LaMBO", "MS_BO"]
SS_ACQFS = ["CArBO", "EIPS", "EI"]


def assert_positive_costs(cost):
    try:
        assert cost.min() > 0
    except AssertionError:
        print("Negative costs detected")


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


def get_initial_data(n, bounds=None, seed=0, acqf=None, params=None):
    X, init_cost = generate_input_data(
        N=n, bounds=bounds, seed=seed, acqf=acqf, params=params
    )
    y = F(X, params).unsqueeze(-1)
    c = Cost_F(X, params)

    if acqf not in MS_ACQFS:
        c = c.sum(dim=1).unsqueeze(-1)

    c_inv = 1 / c.sum(dim=1)
    c_inv = c_inv.to(DEVICE)
    c_inv = c_inv.unsqueeze(-1)

    return X, y, c, c_inv, init_cost


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

    return bounds


def get_random_observations(N=None, bounds=None, seed=None):
    torch.manual_seed(seed=seed)
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
    X = X.to(DEVICE)
    return X


def initialize_GP_model(X, y, params=None):
    X_, y_ = X + 0, y + 0
    X_, y_ = X_.double(), y_.double()
    gp_model = SingleTaskGP(X_, y_).to(X_)
    gp_model = gp_model.to(DEVICE)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(DEVICE)
    return mll, gp_model


def generate_prefix_pool(X, Y, acqf, params):
    first_idx = params["n_init_data"]
    x, y = X[first_idx:], Y[first_idx:]

    data_pool = [(x[i, :], y[i].item()) for i in range(x.shape[0])]
    data_pool.sort(key=lambda d: d[1], reverse=True)
    prefix_pool = [[]]

    if acqf not in ["EEIPU", "EIPU-MEMO"]:
        return prefix_pool

    for i, (param_config, obj) in enumerate(data_pool):
        if i >= params["n_prefixes"]:
            break

        prefix = []
        n_memoizable_stages = len(params["h_ind"]) - 1

        # mem_stages = random.randint(1, n_memoizable_stages)
        for j in range(n_memoizable_stages):
            stage_params = params["h_ind"][j]
            prefix.append(list(param_config[stage_params].cpu().detach().numpy()))
            prefix_pool.append(copy.deepcopy(prefix))

    return prefix_pool
