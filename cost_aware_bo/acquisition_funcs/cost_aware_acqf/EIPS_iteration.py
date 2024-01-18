from .EIPS import EIPS
from cost_aware_bo.functions.processing_funcs import (
    normalize,
    unnormalize,
    standardize,
    unstandardize,
    get_gen_bounds,
)
from cost_aware_bo.functions.iteration_funcs import get_gp_models, get_cost_model
from cost_aware_bo.optimize_mem_acqf import optimize_acqf_by_mem
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective


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
    train_x = normalize(X, bounds=bounds["x"])
    train_y = standardize(y, bounds["y"])

    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)

    norm_bounds = get_gen_bounds(
        params["h_ind"], params["normalization_bounds"], bound_type="norm"
    )

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

    new_x = unnormalize(new_x, bounds=bounds["x"])

    return new_x, n_memoised, acq_value
