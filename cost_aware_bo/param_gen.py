import csv
import json
import os
import random
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import botorch
import numpy as np
import torch
import wandb
import pickle

from .functions.processing_funcs import get_dataset_bounds
from .optimizer.optimize_acqf_funcs import (
    _optimize_acqf_batch,
    gen_batch_initial_conditions,
    gen_candidates_scipy,
    optimize_acqf,
)
from .acquisition_funcs.LaMBO.LaMBO import (
    select_arm,
    update_all_probabilities,
    update_loss_estimators,
    build_partitions,
    get_pdf,
    build_tree,
    remove_invalid_partitions,
)
from .acquisition_funcs.cost_aware_acqf import CArBO_iteration, EIPS_iteration
from .acquisition_funcs.EEIPU.EEIPU_iteration import eeipu_iteration
from .acquisition_funcs.EI.EI_iteration import ei_iteration
from .acquisition_funcs.LaMBO.LaMBO_iteration import lambo_iteration
from .acquisition_funcs.MS_BO.MS_BO_iteration import msbo_iteration

botorch.optim.optimize.optimize_acqf = optimize_acqf
botorch.optim.optimize._optimize_acqf_batch = _optimize_acqf_batch
botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy
botorch.optim.initializers.gen_batch_initial_conditions = gen_batch_initial_conditions

iteration_funcs = {
    "EEIPU_iteration": eeipu_iteration,
    "MS_BO_iteration": msbo_iteration,
    "LaMBO_iteration": lambo_iteration,
    "EI_iteration": ei_iteration,
    "CArBO_iteration": CArBO_iteration.carbo_iteration,
    "MS_CArBO_iteration": CArBO_iteration.carbo_iteration,
    "EIPS_iteration": EIPS_iteration.eips_iteration,
}

# TODO: Define bounds for hyperparameter values that are bounded


def read_json(filename):
    file = open(f"{filename}.json")
    params = json.load(file)
    return params


def sample_value(lower, upper, seed, dtype=float, choice_list=[]):
    np.random.seed(seed)

    assert lower < upper, "hyperparameter lower bound must be less than upper bound"
    assert not (
        dtype and choice_list
    ), "Only one of dtype and choice_list should be set to a value"
    if dtype == float:
        val = np.random.uniform(lower, upper)
    elif dtype == round:
        val = np.random.randint(lower, upper)
    elif dtype == str:
        val = choice_list[np.random.randint(0, len(choice_list))]
    else:
        raise ValueError("Invalid `dtype` provided. Provide one of `int` and `float`")

    return val


def clip(
    data: Union[List, int, float],
    lower: Union[int, float] = None,
    upper: Union[int, float] = None,
):
    if not lower and not upper:
        return data
    if isinstance(data, list):
        return [clip(val, lower, upper) for val in data]
    return max(lower, min(upper, data))


def write_hparams(hparams, filename):
    with open(filename, "w") as f:
        json.dump(hparams, f)


def read_hparams(filename):
    with open(filename) as f:
        return json.load(f)


def write_dataset(dataset):
    with open(HP_DATASET, "w") as f:
        dataset = dataset.copy()
        for key, val in dataset.items():
            dataset[key] = val.tolist()  # if isinstance(val, torch.Tensor) else val
        json.dump(dataset, f, sort_keys=True)


def generate_hparams(
    hp: List[List[Union[int, float]]],
    x_bounds: List[Tuple[Union[int, float]]],
    dtypes: List[str],
    sampling_seed,
):
    """When a new set of hyperparaters, hp, is provided from the bayesian optimizer, this function saves the hyperparameter values for
    each stage in the respective files. When hp is none (i.e. when generating the warm up values before running BO), the function for
    generating the hps for each stage will randomly sample from a range of values

    hp_stg_paths: list of paths where stage hyperparameters are stored. It must be arranged in the run right order
    x_bounds: range of values that each hyperparameter can take. Each list in the main list represents a stage. Each tuple in the stage list
        represents the (lower, upper) bounds of that hyperparameter of that stage.
    dtypes: similar to x_bounds. But it contains strings of either `float` or `int` which stages the data type the corresponding hyperparameter
        takes
    """
    new_hp = []
    hp = hp[:] if hp else hp
    for stg_bounds, stg_dtypes in zip(x_bounds, dtypes):
        stg_hps = []

        for bound, dtype in zip(stg_bounds, stg_dtypes):
            lower, upper = bound
            if dtype == "int":
                dtype = "round"  # casting 1.9999 as int gives 1, which is not ideal. Thus, we change int to round

            if hp:
                val = clip(eval(dtype)(hp.pop(0)), lower, upper)
            else:
                val = sample_value(lower, upper, sampling_seed, dtype=eval(dtype))
            stg_hps.append(val)

        new_hp.extend(stg_hps)

    return new_hp


def read_stg_costs(stage_costs_outputs):
    new_stg_costs = []
    stg_json = {}
    for i in range(len(stage_costs_outputs)):
        if not os.path.exists(stage_costs_outputs[i]):
            if i > 1:
                raise FileNotFoundError(f"Cannot find {stage_costs_outputs[i]}")
            break
        with open(stage_costs_outputs[i]) as stg:
            stg = stg.read().strip()
            stg_json = json.loads(
                stg[: stg.find("}") + 1] if "}" in stg else f"{stg}{'}'}"
            )
            new_stg_costs.append(stg_json["wall_time"])

    return new_stg_costs


def read_objective(obj_output):
    if not Path(obj_output).exists():
        raise FileNotFoundError(f"File {obj_output} not found.")
    with open(obj_output) as f:
        obj = f.read()
        return json.loads(obj).get("AU_ROC", float("inf"))


# def read_hp_dataset(HP_DATASET):
#     if not os.path.exists(HP_DATASET):
#         return {"x":[], "y":[], "c":[]}
#     with open(HP_DATASET) as f:
#         return json.load(f)


def update_dataset_new_run(
    dataset,
    new_hp_dict,
    new_stg_costs,
    new_obj,
    x_bounds,
    acqf,
    dtype=torch.float64,
    device="cuda",
):
    # Check PIRLIB output directory for new objective and cost values
    new_stg_costs = torch.tensor(new_stg_costs, dtype=dtype, device=device)
    new_obj = torch.tensor([new_obj], dtype=dtype, device=device)

    new_stg_costs = new_stg_costs.unsqueeze(0)
    # if acqf == 'EEIPU':
    #     new_stg_costs = new_stg_costs.unsqueeze(0)
    # else:
    #     new_stg_costs = new_stg_costs.sum().unsqueeze(0).unsqueeze(0)

    if dataset.get("y") is not None and dataset.get("c") is not None:
        dataset["y"] = torch.cat([dataset["y"], new_obj])
        dataset["c"] = torch.cat([dataset["c"], new_stg_costs])
    else:
        dataset["y"] = new_obj
        dataset["c"] = new_stg_costs

    # new_hp = torch.tensor([new_hp], device=device, dtype=dtype)
    for hp_name, hp in new_hp_dict.items():
        if dataset.get("x") is None:
            dataset["x"] = {}
        if dataset["x"].get(hp_name) is None:
            dataset["x"][hp_name] = torch.tensor([hp], device=device, dtype=dtype)
        else:
            dataset["x"][hp_name] = torch.cat(
                [dataset["x"][hp_name], torch.tensor([hp], device=device, dtype=dtype)]
            )

    # dataset["x"] = torch.cat([dataset["x"], new_hp_dict])
    x_bounds = torch.cat(
        [
            torch.tensor(stage_bounds, device=device, dtype=dtype)
            for stage_bounds in x_bounds
        ]
    )
    bounds = get_dataset_bounds(dataset["x"], dataset["y"], dataset["c"], x_bounds)
    bounds = {f"{key}_bounds": val for key, val in bounds.items()}
    dataset.update(bounds)
    # write_dataset(dataset)
    # print(f"FOR ACQF = {acqf} THE DATASET SO FAR IS:\n{dataset}")

    return dataset


def log_metrics(
    dataset,
    logging_metadata: Dict,
    exp_name,
    verbose: bool = False,
    iteration=None,
    trial=None,
    acqf=None,
    eta=None,
):
    n_memoised = logging_metadata.pop("n_memoised")
    best_f = dataset["y"].max().item()
    new_y = dataset["y"][-1].item()
    stage_cost_list = dataset["c"][-1, :].tolist()
    sum_stages = sum(stage_cost_list)
    cum_cost = dataset["c"].sum()
    inv_cost = 1 / sum_stages
    dataset_x = [hp_tensor.tolist() for hp_tensor in dataset["x"].values()]

    hp_table = wandb.Table(columns=list(range(len(dataset_x[0]))), data=dataset_x)

    if verbose:  # and iteration >= bo_params['n_init_data']:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", end=":\t")
        print(f"f(x)={new_y:>4.3f}", end="\t")
        print(
            "c(x)=[" + ", ".join("{:.3f}".format(val) for val in stage_cost_list) + "]",
            end="\t",
        )
        print(f"s(c(x)) = [{sum_stages:>4.3f}]", end="\t")
        print(f"c(c) = {cum_cost:>4.3f}", end="\t")
        print(f"num_memoise = {n_memoised}", end="\n\n")
        print("===" * 20)
        print("\n")

    csv_log = dict(
        acqf=acqf,
        trial=trial,
        iteration=iteration,
        best_f=best_f,
        sum_c_x=sum_stages,
        cum_costs=cum_cost,
        eta=eta,
    )
    csv_log_table = wandb.Table(columns=list(csv_log.keys()))
    csv_log_table.add_data(*csv_log.values())

    log = dict(
        best_f=best_f,
        f_x=new_y,
        sum_c_x=sum_stages,
        cum_costs=cum_cost,
        inv_cost=inv_cost,
        c_x=dict(zip(map(str, range(len(stage_cost_list))), stage_cost_list)),
        hp_table=hp_table,
        csv_log_table=csv_log_table,
    )

    dir_name = f"./experiment_logs/{exp_name}"
    Path(dir_name).mkdir(exist_ok=True, parents=True)  # Create if it doesn't exist
    csv_file_name = f"{dir_name}/{acqf}_trial_{trial}.csv"

    # Check if the file exists
    try:
        with open(csv_file_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            fieldnames = next(reader)  # Read the headers in the first row

    except FileNotFoundError:
        # If file does not exist, create it and write headers
        fieldnames = [
            "acqf",
            "trial",
            "iteration",
            "best_f",
            "sum_c_x",
            "cum_costs",
            "eta",
        ]
        with open(csv_file_name, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Append data
    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(csv_log)

    wandb.log(log, step=iteration)
    return


def lambo_preprocessing(
    acqf, h_ind, x_bounds, n_stages, trial, first_iter, iteration, exp_name
):
    # TODO: Refactor this if block to take it outside the function
    if acqf != "LaMBO" or iteration < first_iter:
        return None, None, None, None, None, None, x_bounds

    tree = None
    tree_path = Path(f"{exp_name}/{acqf}/tree.pickle_{trial}.pkl")
    if tree_path.exists():
        with open(tree_path, "rb") as tree_file:
            root, mset = pickle.load(tree_file)
            probs, loss, h, global_input_bounds, arm_idx = root.retrieve_data()
            return root, mset, loss, probs, arm_idx, h, global_input_bounds

    n_leaves = 2 ** (n_stages - 1)
    probs = get_pdf(n_leaves)

    partitions, last_stage_partition = build_partitions(x_bounds, h_ind, n_stages)

    depths = [1 for stage in range(n_stages - 1)]

    mset, root = build_tree(partitions, depths, last_stage_partition)
    arm_idx = random.randint(0, n_leaves)

    H = sum(depths)
    h = H + 0

    loss = np.zeros([n_leaves, H])
    return root, mset, loss, probs, arm_idx, h, x_bounds


def lambo_pre_iteration(
    mset, root, acqf, probs, h, n_stages, arm_idx, bounds, first_iter, iteration
):
    # TODO: Refactor this if block to take it outside the function
    if acqf != "LaMBO" or iteration < first_iter:
        return bounds, 0

    n_leaves = 2 ** (n_stages - 1)
    leaf_bounds = mset.leaves
    input_bounds, arm_idx = select_arm(root, leaf_bounds, probs, h, arm_idx, n_leaves)
    bounds["x"] = input_bounds

    return bounds, arm_idx


def lambo_post_iteration(
    acqf,
    root,
    mset,
    loss,
    probs,
    arm_idx,
    acq_value,
    n_stages,
    global_input_bounds,
    h_ind,
    first_iter,
    iteration,
    trial,
    exp_name,
):
    # TODO: Refactor this if block to take it outside the function
    if acqf != "LaMBO" or iteration < first_iter:
        return

    n_leaves = 2 ** (n_stages - 1)
    depths = [1 for stage in range(n_stages - 1)]
    H = sum(depths)

    sigma = np.array(random.choices([-1, 1], k=H))
    sigma[-1] = -1

    h = np.where(sigma == -1)[0][0]

    loss = update_loss_estimators(loss, root, probs, arm_idx, sigma, H, acq_value)

    probs = update_all_probabilities(loss, probs, arm_idx, n_leaves)

    # Probabilities are reinitialized if an arm is invalidated
    global_input_bounds, probs, loss = remove_invalid_partitions(
        global_input_bounds,
        probs,
        loss,
        h_ind,
        n_leaves,
        H,
        n_stages,
        mset.leaf_partitions,
    )

    partitions, last_stage_partition = build_partitions(
        global_input_bounds, h_ind, n_stages
    )

    mset, root = build_tree(partitions, depths, last_stage_partition)

    root.save_data(probs, loss, h, global_input_bounds, arm_idx)

    tree_path = Path(f"{exp_name}/{acqf}/tree.pickle_{trial}.pkl")
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tree_path, "wb") as file:
        tree = (root, mset)
        pickle.dump(tree, file)

    return mset, root


def reformat_xbounds(x_bounds, device="cuda"):
    x_b = [[], []]
    for stage in x_bounds:
        for param_bounds in stage:
            x_b[0].append(param_bounds[0])
            x_b[1].append(param_bounds[1])
    return torch.tensor(x_b, device=device)


def generate_hps(
    dataset,
    hp_sampling_range,
    iteration,
    params,
    exp_name,
    trial,
    consumed_budget=None,
    acq_type="EEIPU",
):
    h_ind = {}  # {0: [0,1], 1: [2,3,4], } [[0,1], [2,3,4]...] [0,1,2,3,4...]
    hp_names = {}  # {0: ["mean", "std"], 1: ["lr", "batch_size", "decay"], }
    stage_ids = sorted(
        [(key.split("__")) for key in hp_sampling_range], key=lambda item: item[0]
    )

    for i, (stg_id, hp_name) in enumerate(stage_ids):
        stg_id = int(stg_id)
        if h_ind.get(stg_id) is None:
            h_ind[stg_id] = [i]
            hp_names[stg_id] = [hp_name]
        else:
            h_ind[stg_id].append(i)
            hp_names[stg_id].append(hp_name)

    h_ind_list = list(dict(sorted(h_ind.items())).values())
    params["h_ind"] = h_ind_list

    x_bounds = {}
    hp_dtypes = {}
    for key, value in hp_sampling_range.items():
        bounds = (value["lower"], value["upper"])
        id = int(key.split("__")[0])
        if x_bounds.get(id) is None:
            x_bounds[id] = [bounds]
            hp_dtypes[id] = [value["dtype"]]
        else:
            x_bounds[id].append(bounds)
            hp_dtypes[id].append(value["dtype"])

    x_bounds = list(dict(sorted(x_bounds.items())).values())
    hp_dtypes = list(dict(sorted(hp_dtypes.items())).values())

    rand_seed = params["rand_seed"]
    torch.manual_seed(seed=rand_seed)
    random.seed(rand_seed)
    botorch.utils.sampling.manual_seed(seed=rand_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_b = reformat_xbounds(x_bounds, device=device)

    first_iter, n_stages = params["n_init_data"] + 1, len(h_ind_list)
    root, mset, loss, probs, arm_idx, h, x_b = lambo_preprocessing(
        acq_type, h_ind_list, x_b, n_stages, trial, first_iter, iteration, exp_name
    )

    new_hp, n_memoised, n_init_data = None, 0, params["n_init_data"]
    if iteration > n_init_data:
        x = torch.stack(list(dataset["x"].values()), axis=1)
        y = dataset["y"].unsqueeze(-1)
        c = dataset["c"]

        bounds = {
            key.replace("_bounds", ""): val
            for key, val in dataset.items()
            if key.endswith("_bounds")
        }

        bounds, arm_idx = lambo_pre_iteration(
            mset,
            root,
            acq_type,
            probs,
            h,
            n_stages,
            arm_idx,
            bounds,
            first_iter,
            iteration,
        )

        bo_iter_function = iteration_funcs[f"{acq_type}_iteration"]
        new_hp, n_memoised, acq_value = bo_iter_function(
            X=x,
            y=y,
            c=c,
            bounds=bounds,
            acqf_str=acq_type,
            decay=params["init_eta"],
            iter=iteration,
            consumed_budget=consumed_budget,
            params=params,
        )
        new_hp = new_hp.squeeze().tolist()

        lambo_post_iteration(
            acq_type,
            root,
            mset,
            loss,
            probs,
            arm_idx,
            acq_value,
            n_stages,
            x_b,
            h_ind_list,
            first_iter,
            iteration,
            trial,
            exp_name,
        )

    # When new_hp is None, `generate_hparams` will generate random samples.
    # It also saves the new_hp to the respective files where the main function can read them
    new_hp = generate_hparams(
        new_hp, x_bounds, hp_dtypes, sampling_seed=iteration * (params["trial"] + 1)
    )

    logging_metadata = {
        "n_memoised": n_memoised,
        "x_bounds": x_bounds,
    }

    # log_metrics(dataset, logging_metadata)

    # Convert new_hp into a dictionary of key representing the stage id and name in this format, "0__mean"
    new_hp_out = {}
    # print("hp_names", hp_names)
    for stage, stg_hps_dict in hp_names.items():
        for hp_name in stg_hps_dict:
            new_hp_out[f"{stage}__{hp_name}"] = new_hp.pop(0)
    return new_hp_out, logging_metadata


if __name__ == "__main__":
    t = time.localtime()
    parser = ArgumentParser()
    parser.add_argument("--trial", type=int, help="The trial number", default=0)
    # parser.add_argument("--iter", type=int, help="The iteration count, starting from 0. Pass -1 to only read the final objective and cost values.", required=True)
    parser.add_argument(
        "--stage_costs_outputs",
        nargs="+",
        help="The output directories of the stages with tunable hps in order.",
        required=True,
    )
    parser.add_argument(
        "--obj-output", help="The output directory of the objective.", required=True
    )
    parser.add_argument(
        "--base-dir", type=Path, help="Base tuun directory.", default=Path(".")
    )
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=3)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument(
        "--acqf",
        type=str,
        help="Acquisition function",
        choices=["EEIPU", "EIPU", "EIPU-MEMO", "EI", "RAND"],
        default="EEIPU",
    )

    args = parser.parse_args()

    BASE_DIR = args.base_dir
    HP_DATASET = BASE_DIR / "hp_dataset.json"

    hp_stg_paths = [
        BASE_DIR / "hparams/stage1.json",
        BASE_DIR / "hparams/stage2.json",
        BASE_DIR / "hparams/stage3.json",
    ]
    # hp_stg_paths_ = [BASE_DIR / "hparams/stage_1.json", BASE_DIR / "hparams/stage_2.json"]

    # args.log_dir.mkdir(parents=True, exist_ok=True)
    stage_costs_outputs = {
        ind: file for ind, file in enumerate(args.stage_costs_outputs)
    }

    # if args.iter==0:
    files_to_delete = [HP_DATASET]
    files_to_delete.extend(stage_costs_outputs.values())
    for file in files_to_delete:
        if os.path.exists(file):
            print("Removing file", file)
            os.remove(file)
    print("All files removed")

    params = read_json("params")
    args_dict = deepcopy(vars(args))
    params.update(args_dict)

    trial = args.trial
    wandb.init(
        entity="cost-bo",
        project="jan-2024-cost-aware-bo",
        group=f"Stacking-{args.exp_group}|-acqf_{args.acqf}|-dec-fac_{args.decay_factor}"
        f"|init-eta_{args.init_eta}",
        name=f"{time.strftime('%Y-%m-%d-%H%M')}-trial-number_{trial}",
        config=params,
    )

    torch.manual_seed(seed=params["rand_seed"])
    # np.random.seed(params['rand_seed'])
    # random.seed(params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=params["rand_seed"])

    for iter in range(1, 2):
        # for iter in range(20):
        hp = generate_hps(
            iteration=iter,
            trial_number=args.trial,
            stage_costs_outputs=stage_costs_outputs,
            obj_output=args.obj_output,
            # config_file=args.config_file,
            hp_stg_paths=hp_stg_paths,
            acq_type=args.acqf,
        )

    print("Done!!!")
