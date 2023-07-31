import json
import random
import subprocess
import sys
import time
from copy import deepcopy
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import yaml

sys.path.append("../codeformer/")

import botorch
import numpy as np
import torch
import wandb
from basicsr.utils.misc import get_time_str
from basicsr.utils.options import parse
from functions import get_dataset_bounds
from optimizer.optimize_acqf_funcs import (_optimize_acqf_batch,
                                           gen_batch_initial_conditions,
                                           gen_candidates_scipy, optimize_acqf)
from single_iteration import bo_iteration, read_json


# TODO: Define bounds for hyperparameter values that are bounded
def sample_value(
    lower, upper, dtype=float, choice_list=[]
):
    assert not(dtype and choice_list), "Only one of dtype and choice_list should be set to a value"
    if dtype==float:
        val = np.random.random()
        val = val * (upper - lower) + lower
    elif dtype==int:
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


def generate_update_stage_hparams(hp: List, hp_path):
    hp_path = Path(hp_path)
    with hp_path.open("r") as f:
        opt: OrderedDict = parse(hp_path, "../codeformer")
        
    optimizable: OrderedDict = opt.get("optimizable")
    hparams: OrderedDict = opt["train"]
    
    stg_hp, stg_bounds = [], []
    hparams_out = {}
    for key1, value1 in optimizable.items():
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                lower, upper = value2
        else:
            lower, upper = value2
        
        dtype = type(lower)
        if hp:
            val = clip(dtype(hp.pop(0)), lower, upper)
        else:
            val = sample_value(lower, upper, dtype=dtype)
        
        stg_hp.append(val)
        stg_bounds.append([lower, upper])
        if isinstance(value1, dict):
            hparams[key1][key2] = val
        else:
            hparams[key1] = val
                
    opt["train"] = hparams
    
    # 
    hp_out_path = OPTIONS_OUT / hp_path.name[hp_path.name.find("stage"):]
    hp_out_path.parent.mkdir(parents=True, exist_ok=True) if hp_out_path.name=="stage1.yml" else None
    opt_dict = json.loads(json.dumps(opt))
    with hp_out_path.open("w") as f:
        yaml.dump(opt_dict, f, sort_keys=False)
    return stg_hp, stg_bounds, hp


def generate_hparams(hp:List=None, hp_stg_paths:List=None):
    '''When a new set of hyperparaters, hp, is provided from the bayesian optimizer, this function saves the hyperparameter values for
    each stage in the respective files. When hp is none (i.e. when generating the warm up values before running BO), the function for 
    generating the hps for each stage will randomly sample from a range of values
    
    hp_stg_paths: list of paths where stage hyperparameters are stored. It must be arranged in the run right order
    '''
    new_x, bounds = [], []
    hp = hp[:] if hp else hp
    for stg_idx, stg_path in enumerate(hp_stg_paths):
        stg_hp, stg_bounds, hp = generate_update_stage_hparams(hp, stg_path)
        new_x.extend(stg_hp)
        bounds.append(stg_bounds)
        # hp_out_paths.append(hp_out_path)
        
    return new_x, list(zip(*bounds))


def read_stg_costs(stage_costs_outputs):
    # stage_costs = ["stage1_cost.json", "stage2_cost.json", "stage3_cost.json"]
    new_stg_costs  = [] 
    for i in range(len(stage_costs_outputs)):
        cost = stage_costs_outputs[i]
        if not cost.exists():
            print(f"Cannot find {cost}")
            if i>1:
                raise FileNotFoundError(f"Cannot find {cost}")
            break 
        with cost.open() as stg:
            stg_json = json.load(stg)
            new_stg_costs.append(stg_json["COST"])
            
    return new_stg_costs

def read_objective(obj_path):
    # obj_path = Path(METRICS_PATH)/"stage3_obj.json"
    if not obj_path.exists():
        raise FileNotFoundError(f"File {obj_path} not found.")
    with obj_path.open() as f:
        return json.load(f)["OBJ"]

def read_hp_dataset(hp_dataset_path):
    # hp_dataset_path = Path(METRICS_PATH)/"hp_dataset.json"
    if not hp_dataset_path.exists():
        return {"x":[], "y":[], "c":[]}
    with hp_dataset_path.open() as f:
        return json.load(f)

def write_dataset(dataset):
    with open(HP_DATASET, "w") as f:
        dataset = dataset.copy()
        for key, val in dataset.items():
            dataset[key] = val.tolist()# if isinstance(val, torch.Tensor) else val
        json.dump(dataset, f, sort_keys=True)

def generate_hps(
    iteration,
    trial_number=1, 
    acq_type="EEIPU", 
    stage_costs_outputs:Dict[int, str]={
        0:"./stage_zero_cost.json", 1:"./stage_one_cost.json", 2:"./stage_two_cost.json",
    },
    obj_output: str = "./score.json",
    hp_stg_paths = [], # The number of paths provided should match the number of stages
    # config_file:Path = Path("")
):
    tic = time.time()
    bo_params = read_json(BASE_DIR / TUUN_DIR / "params")
    dtype = torch.double
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed=bo_params['rand_seed'])
    random.seed(bo_params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=bo_params['rand_seed'])

    dataset = read_hp_dataset(HP_DATASET)
    dataset = {key:torch.tensor(val, device=device, dtype=dtype) for key, val in dataset.items()}
    
    new_x, y_pred, n_memoised, E_c, E_inv_c = None, None, 0, None, None
    if iteration >= bo_params['n_init_data']:
        bounds = {key.replace("_bounds", ""):val for key,val in dataset.items() if key.endswith("_bounds")}
        new_x, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(X=dataset["x"], y=dataset["y"].unsqueeze(-1), c=dataset["c"], bounds=bounds, acqf_str=acq_type, decay=bo_params['init_eta'], iter=iteration, params=bo_params)
        new_x = new_x.squeeze().tolist()
    
    # When new_x is None, `generate_hparams` will generate random samples.
    # It also saves the new_x to the respective files where the main function can read them 
    new_x, x_bounds = generate_hparams(new_x, hp_stg_paths)
    
    ## Execute model (PIRLib)
    
    # resp = subprocess.run(["python", "stacking_algo.py"], capture_output=True, text=True) # For testing
    keys = stage_costs_outputs.keys()
    # stage_costs_outputs_list = [stage_costs_outputs[i] for i in keys]
    # print("stage_costs_outputs_list", stage_costs_outputs_list)
    
    cmd = ["bash", "codeformer/run_codeformer.sh", EXP_NAME, METRICS_PATH]
    # cmd = ["argo", "submit", "-n", "gpu-14", "--wait", config_file],
    cmd.append("--disable-cache") if acq_type=="EI" else None
    resp = subprocess.run(cmd, capture_output=True, text=True)
    print(resp.stdout)
    print("Done running model")
    print(resp.stderr)
    # if resp.stderr:
    #     raise Exception(resp.stderr)
    # print(resp.stdout)
    while not Path(obj_output).exists():
        print("Sleeping for 1 sec...")
        time.sleep(1)
    
    def update_dataset_new_run(dataset, new_x, stage_costs_outputs, obj_output, x_bounds, dtype=torch.float64, device="cuda"):
        # Check PIRLIB output directory for new objective and cost values
        new_stg_costs = read_stg_costs(stage_costs_outputs)
        new_stg_costs = torch.tensor(new_stg_costs, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
        new_obj = read_objective(obj_output)
        new_obj = torch.tensor([new_obj],dtype=dtype, device=device)
                
        new_x = torch.tensor([new_x], device=device, dtype=dtype)
        
        dataset["x"] = torch.cat([dataset["x"], new_x])
        dataset["y"] = torch.cat([dataset['y'], new_obj])
        dataset["c"] = torch.cat([dataset['c'], new_stg_costs], axis=1)

        bounds = get_dataset_bounds(dataset["x"], dataset["y"], dataset["c"], torch.tensor(x_bounds, device=device, dtype=dtype))
        bounds = {f"{key}_bounds":val for key, val in bounds.items()}
        dataset.update(bounds)
        
        write_dataset(dataset)
        
        return dataset
    
    dataset = update_dataset_new_run(dataset, new_x, stage_costs_outputs, obj_output, x_bounds, dtype, device)
    best_f = dataset["y"].max().item()
    new_y = dataset["y"][-1].item()
    stage_cost_list = dataset["c"][:, -1].squeeze()
    sum_stages = sum(stage_cost_list)
    cum_cost = dataset["c"].sum()
    inv_cost = 1/sum_stages
    dataset_x = dataset["x"]
    hp_table = wandb.Table(columns=list(range(len(dataset_x[0]))), data=dataset_x.tolist())
    
    print(f"\n[{time.strftime('%Y-%m-%d-%H%M')}] Iteration-{iteration} [{acq_type}] Trial No. #{trial_number} Runtime: {time.time()-tic}")
    if bo_params['verbose']: # and iteration >= bo_params['n_init_data']:
        print(f"f(x^)={y_pred}", end="   ")
        print(f"f(x)={new_y:>4.3f}", end="   ")
        print(f"c(x)=[" + ', '.join('{:.3f}'.format(val) for val in stage_cost_list) + "]", end="   ")
        print(f"s(c(x)) = [{sum_stages:>4.3f}]", end="   ")
        print(f"c(c) = {cum_cost:>4.3f}", end="   ")
        print(f"num_memoise = {n_memoised}", end="\n\n")
        print("==="*20)
        print("\n")
    
  
    # with (log_dir/f"trial_{trial_number}_log.jsonl").open("a") as f:
    #     f.write(json.dumps(log_vals)+"\n")
    
    # if iteration >= bo_params['n_init_data']:    
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
            eta=bo_params['init_eta'],
            hp_table=hp_table,
        )
    wandb.log(log)
    return new_x

if __name__=="__main__":
    t = time.localtime()
    parser = ArgumentParser()
    parser.add_argument("--trial", type=int, help="The trial number", default=0)
    # parser.add_argument("--iter", type=int, help="The iteration count, starting from 0. Pass -1 to only read the final objective and cost values.", required=True)
    # parser.add_argument("--stage_costs_outputs", nargs='+', help="The output directories of the stages with tunable hps in order.", required=True)
    # parser.add_argument("--obj-output", help="The output directory of the objective.", required=True)
    parser.add_argument("--base-dir", type=Path, help="Base tuun directory.", default=Path("."))
    parser.add_argument("--exp-name", type=Path, help="Experiment name.")
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=3)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPU", "EIPU-MEMO", "EI", "RAND"], default="EEIPU")

    args = parser.parse_args()
    
    TUUN_DIR = "cost-aware-bo"
        
    BASE_DIR = args.base_dir
    EXP_NAME = args.exp_name
    METRICS_PATH = BASE_DIR / TUUN_DIR / "metrics" / EXP_NAME
    HP_DATASET = METRICS_PATH / "hp_dataset.json"
    
    options = "codeformer/options"
    OPTIONS_OUT = BASE_DIR / options / EXP_NAME
    hp_stg_paths = [
        BASE_DIR / options / "VQGAN_512_ds32_nearest_stage1.yml",
        BASE_DIR / options / "CodeFormer_stage2.yml",
        BASE_DIR / options / "CodeFormer_stage3.yml"
    ]
    
    stages = ["stage1_cost.json", "stage2_cost.json", "stage3_cost.json"]
    stage_costs_outputs = {ind:METRICS_PATH/file for ind,file in enumerate(stages)}
    
    obj_output = METRICS_PATH / "stage3_obj.json"
    
    params = read_json(BASE_DIR / TUUN_DIR / "params")
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
    botorch.optim.optimize.optimize_acqf = optimize_acqf
    botorch.optim.optimize._optimize_acqf_batch = _optimize_acqf_batch
    botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy
    botorch.optim.initializers.gen_batch_initial_conditions = gen_batch_initial_conditions
    
    
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
            obj_output=obj_output,
            # config_file=args.config_file,
            hp_stg_paths=hp_stg_paths,
            acq_type=args.acqf,
        )        
    print("Done!!!")
    