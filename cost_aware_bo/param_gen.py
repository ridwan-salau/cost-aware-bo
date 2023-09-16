import copy
import json
import random
import os
from argparse import ArgumentParser
import time
from typing import List, Union, Dict, Tuple
from pathlib import Path
import subprocess
from copy import deepcopy

import botorch
import torch
import numpy as np
import wandb
from einops import rearrange

from .single_iteration import bo_iteration
from .optimizer.optimize_acqf_funcs import optimize_acqf, _optimize_acqf_batch, gen_candidates_scipy, gen_batch_initial_conditions
from .functions import get_dataset_bounds

botorch.optim.optimize.optimize_acqf = optimize_acqf
botorch.optim.optimize._optimize_acqf_batch = _optimize_acqf_batch
botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy
botorch.optim.initializers.gen_batch_initial_conditions = gen_batch_initial_conditions

# TODO: Define bounds for hyperparameter values that are bounded

def read_json(filename):
    file = open(f"{filename}.json")
    params = json.load(file)
    return params


def sample_value(
    lower, upper, dtype=float, choice_list=[]
):
    assert lower < upper, "hyperparameter lower bound must be less than upper bound"
    assert not(dtype and choice_list), "Only one of dtype and choice_list should be set to a value"
    if dtype==float:
        val = np.random.random()
        val = val * (upper - lower) + lower
    elif dtype==round:
        val = np.random.randint(lower, upper)
    elif dtype==str:
        val = choice_list[np.random.randint(0, len(choice_list))]
    else:
        raise ValueError("Invalid `dtype` provided. Provide one of `int` and `float`")
    
    return val
    
def clip(data:Union[List, int, float], lower:Union[int, float]=None, upper:Union[int, float]=None):
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
            dataset[key] = val.tolist()# if isinstance(val, torch.Tensor) else val
        json.dump(dataset, f, sort_keys=True)
    

def generate_hparams(hp: List[List[int|float]], x_bounds: List[Tuple[int|float]], dtypes: List[str]):
    '''When a new set of hyperparaters, hp, is provided from the bayesian optimizer, this function saves the hyperparameter values for
    each stage in the respective files. When hp is none (i.e. when generating the warm up values before running BO), the function for 
    generating the hps for each stage will randomly sample from a range of values
    
    hp_stg_paths: list of paths where stage hyperparameters are stored. It must be arranged in the run right order
    x_bounds: range of values that each hyperparameter can take. Each list in the main list represents a stage. Each tuple in the stage list 
        represents the (lower, upper) bounds of that hyperparameter of that stage.
    dtypes: similar to x_bounds. But it contains strings of either `float` or `int` which stages the data type the corresponding hyperparameter
        takes
    '''
    new_hp = []
    hp = hp[:] if hp else hp
    for stg_bounds, stg_dtypes in zip(x_bounds, dtypes):
        stg_hps = []
        
        for bound, dtype in zip(stg_bounds, stg_dtypes):
            lower, upper = bound
            if dtype=="int": dtype="round" # casting 1.9999 as int gives 1, which is not ideal. Thus, we change int to round
            
            if hp:
                val = clip(eval(dtype)(hp.pop(0)), lower, upper)
            else:
                val = sample_value(lower, upper, dtype=eval(dtype))
            stg_hps.append(val)
        
        new_hp.extend(stg_hps)

    return new_hp
    
def read_stg_costs(stage_costs_outputs):
    new_stg_costs  = [] 
    stg_json = {}
    for i in range(len(stage_costs_outputs)):
        if not os.path.exists(stage_costs_outputs[i]):
            if i>1:
                raise FileNotFoundError(f"Cannot find {stage_costs_outputs[i]}")
            break 
        with open(stage_costs_outputs[i]) as stg:
            stg = stg.read().strip()
            stg_json = json.loads(stg[:stg.find("}")+1] if "}" in stg else f"{stg}{'}'}")
            new_stg_costs.append(stg_json["wall_time"])
            
    return new_stg_costs

def read_objective(obj_output):
    if not Path(obj_output).exists():
        raise FileNotFoundError(f"File {obj_output} not found.")
    with open(obj_output) as f:
        obj = f.read()
        return json.loads(obj).get("AU_ROC",float("inf"))

# def read_hp_dataset(HP_DATASET):
#     if not os.path.exists(HP_DATASET):
#         return {"x":[], "y":[], "c":[]}
#     with open(HP_DATASET) as f:
#         return json.load(f)


def update_dataset_new_run(dataset, new_hp_dict, new_stg_costs, new_obj, x_bounds, dtype=torch.float64, device="cuda"):
    # Check PIRLIB output directory for new objective and cost values
    new_stg_costs = torch.tensor(new_stg_costs, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
    new_obj = torch.tensor([new_obj],dtype=dtype, device=device)
    
    if dataset.get("y") is not None and dataset.get("c") is not None:
        dataset["y"] = torch.cat([dataset['y'], new_obj])
        dataset["c"] = torch.cat([dataset['c'], new_stg_costs], axis=1)
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
            dataset["x"][hp_name] = torch.cat([dataset["x"][hp_name], torch.tensor([hp], device=device, dtype=dtype)])
    
    # dataset["x"] = torch.cat([dataset["x"], new_hp_dict])
    x_bounds = torch.cat([torch.tensor(stage_bounds, device=device, dtype=dtype) for stage_bounds in x_bounds])
    bounds = get_dataset_bounds(dataset["x"], dataset["y"], dataset["c"], x_bounds)
    bounds = {f"{key}_bounds":val for key, val in bounds.items()}
    dataset.update(bounds)
    
    # write_dataset(dataset)
    
    return dataset

def log_metrics(dataset, logging_metadata: Dict, verbose: bool=False):
    y_pred = logging_metadata.pop("y_pred")
    n_memoised = logging_metadata.pop("n_memoised")
    E_inv_c = logging_metadata.pop("E_inv_c")
    E_c = logging_metadata.pop("E_c")
    
    # dataset = update_dataset_new_run(dataset, new_hp, stage_costs_outputs, obj_output, x_bounds, dtype, device)
    best_f = dataset["y"].max().item()
    new_y = dataset["y"][-1].item()
    stage_cost_list = dataset["c"][:, -1].squeeze()
    sum_stages = sum(stage_cost_list)
    cum_cost = dataset["c"].sum()
    inv_cost = 1/sum_stages
    dataset_x = [hp_tensor.tolist() for hp_tensor in dataset["x"].values()]
    columns = list(dataset["x"].keys())
    
    hp_table = wandb.Table(columns=list(range(len(dataset_x[0]))), data=dataset_x)
    
    if verbose: # and iteration >= bo_params['n_init_data']:
        print(f"f(x^)={y_pred}", end="   ")
        print(f"f(x)={new_y:>4.3f}", end="   ")
        print(f"c(x)=[" + ', '.join('{:.3f}'.format(val) for val in stage_cost_list) + "]", end="   ")
        print(f"s(c(x)) = [{sum_stages:>4.3f}]", end="   ")
        print(f"c(c) = {cum_cost:>4.3f}", end="   ")
        print(f"num_memoise = {n_memoised}", end="\n\n")
        print("==="*20)
        print("\n")
    
    log = dict(
            best_f=best_f,
            f_hat_x=y_pred,
            f_x=new_y,
            f_res=abs(y_pred - new_y) if y_pred else None,
            sum_c_x=sum_stages,
            cum_costs=cum_cost,
            E_inv_c=E_inv_c,
            sum_Ec=sum(E_c) if E_c else None,
            inv_cost=inv_cost,
            E_c=dict(zip(map(str,range(len(E_c))) ,E_c)) if E_c else None,
            c_x=dict(zip(map(str,range(len(stage_cost_list))) ,stage_cost_list)),
            c_res=dict(zip(map(str,range(len(stage_cost_list))) ,[abs(act-est) for act, est in zip(E_c,stage_cost_list)])) if E_c else None,
            inv_c_res=abs(E_inv_c-inv_cost) if E_inv_c else None,
            hp_table=hp_table,
        )
    wandb.log(log)
    return


def generate_hps(
    dataset,
    hp_sampling_range,
    iteration,
    params,
    acq_type="EEIPU", 
):
    dtype = torch.double
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    
    h_ind = {}      # {0: [0,1], 1: [2,3,4], }
    hp_names = {}    # {0: ["mean", "std"], 1: ["lr", "batch_size", "decay"], }
    stage_ids = sorted([(key.split("__")) for key in hp_sampling_range], key=lambda item: item[0])
    
    for i, (stg_id, hp_name) in enumerate(stage_ids):
        stg_id = int(stg_id)
        if h_ind.get(stg_id) is None:
            h_ind[stg_id] = [i]
            hp_names[stg_id] = [hp_name]
        else:
            h_ind[stg_id].append(i)
            hp_names[stg_id].append(hp_name)
    
    params["h_ind"] = list(dict(sorted(h_ind.items())).values())
    
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
    
    new_hp, y_pred, n_memoised, E_c, E_inv_c = None, None, 0, None, None
    if iteration >= params["n_init_data"]:
        # Convert to tensors
        # print(dataset)
        x = torch.stack(list(dataset["x"].values()), axis=1)
        y = dataset["y"].unsqueeze(-1)
        c = dataset["c"]
        
        bounds = {key.replace("_bounds", ""):val for key,val in dataset.items() if key.endswith("_bounds")}
        new_hp, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(
            X=x, y=y, c=c, bounds=bounds, acqf_str=acq_type, decay=params["init_eta"], iter=iteration, params=params)
        new_hp = new_hp.squeeze().tolist()
    
    # When new_hp is None, `generate_hparams` will generate random samples.
    # It also saves the new_hp to the respective files where the main function can read them 
    new_hp = generate_hparams(new_hp, x_bounds, hp_dtypes)
    
    logging_metadata = {"n_memoised": n_memoised, "y_pred": y_pred, "E_c": E_c, "E_inv_c": E_inv_c, "x_bounds": x_bounds}
    
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
    parser.add_argument("--stage_costs_outputs", nargs='+', help="The output directories of the stages with tunable hps in order.", required=True)
    parser.add_argument("--obj-output", help="The output directory of the objective.", required=True)
    parser.add_argument("--base-dir", type=Path, help="Base tuun directory.", default=Path("."))
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=3)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPU", "EIPU-MEMO", "EI", "RAND"], default="EEIPU")

    args = parser.parse_args()
    
    BASE_DIR = args.base_dir
    HP_DATASET = BASE_DIR / "hp_dataset.json"
    
    hp_stg_paths = [BASE_DIR / "hparams/stage1.json", BASE_DIR / "hparams/stage2.json", BASE_DIR / "hparams/stage3.json"]
    # hp_stg_paths_ = [BASE_DIR / "hparams/stage_1.json", BASE_DIR / "hparams/stage_2.json"]
    
    # args.log_dir.mkdir(parents=True, exist_ok=True)
    stage_costs_outputs = {ind:file for ind,file in enumerate(args.stage_costs_outputs)}
    
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
        project="memoised-realworld-exp",
        group=f"Stacking-{args.exp_group}|-acqf_{args.acqf}|-dec-fac_{args.decay_factor}"
                f"|init-eta_{args.init_eta}",
        name=f"{time.strftime('%Y-%m-%d-%H%M')}-trial-number_{trial}",
        config=params
    )
    
    
    
    torch.manual_seed(seed=params['rand_seed'])
    # np.random.seed(params['rand_seed'])
    # random.seed(params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=params['rand_seed'])
    
    for iter in range(1, params["n_iters"]+1):
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