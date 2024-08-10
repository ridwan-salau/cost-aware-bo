import csv

import torch
from botorch import fit_gpytorch_model

from cost_aware_bo.functions.processing_funcs import (
    assert_positive_costs,
    initialize_GP_model,
    standardize,
)


def iteration_logs(acqf, trial_number, iteration, best_f, sum_stages, cum_cost):
    log = dict(
        acqf=acqf,
        trial=trial_number,
        iteration=iteration,
        best_f=best_f,
        sum_c_x=sum_stages,
        cum_costs=cum_cost,
    )

    dir_name = "syn_logs_"
    csv_file_name = f"{dir_name}/{acqf}_trial_{trial_number}.csv"

    try:
        with open(csv_file_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            fieldnames = next(reader)

    except FileNotFoundError:
        fieldnames = ["acqf", "trial", "iteration", "best_f", "sum_c_x", "cum_costs"]
        with open(csv_file_name, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(log)


def get_gp_models(X, y, iter, params=None):
    mll, gp_model = initialize_GP_model(X, y, params=params)
    fit_gpytorch_model(mll)
    return mll, gp_model


def get_multistage_cost_models(X, C, iter, param_idx, bounds, acqf):
    cost_mll, cost_gp = [], []
    for i in range(C.shape[1]):
        stage_cost = C[:, i].unsqueeze(-1)

        assert_positive_costs(stage_cost)

        log_sc = torch.log(stage_cost)

        norm_stage_cost = standardize(log_sc, bounds["c"][:, i])
        stage_idx = param_idx[i]
        stage_x = X[:, stage_idx] + 0

        stage_mll, stage_gp = get_gp_models(stage_x, norm_stage_cost, iter)

        cost_mll.append(stage_mll)
        cost_gp.append(stage_gp)
    return cost_mll, cost_gp


def get_cost_model(X, C, iter, param_idx, bounds, acqf):
    assert_positive_costs(C)

    log_sc = torch.log(C)

    norm_cost = standardize(log_sc, bounds["c"][:, 0])

    x = X + 0

    cost_mll, cost_gp = get_gp_models(x, norm_cost, iter)

    return [cost_mll], [cost_gp]
