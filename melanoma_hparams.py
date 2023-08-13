import copy
import json
import random
import os
from argparse import ArgumentParser
import time
from typing import List, Union, Dict
from pathlib import Path
import subprocess
from copy import deepcopy

import botorch
import torch
import numpy as np
import wandb

from single_iteration import bo_iteration, read_json
from optimizer.optimize_acqf_funcs import optimize_acqf, _optimize_acqf_batch, gen_candidates_scipy, gen_batch_initial_conditions
from functions import get_dataset_bounds

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
    
def generate_update_stage_hparams(hp: List, hp_stg, stage_idx):
    
    stg_hp, stg_bounds = [], []
    # hparams_out = {}
    # for sub_stage, sub_hparams in hparams[stage_idx].items():
    #     tunable_param_keys = list(filter(lambda k: k.startswith("__"), sub_hparams))
    #     main_params = dict(filter(lambda k: not k[0].startswith("__"), sub_hparams.items()))
    for key, val in hp_stg.items():
        if val.get("range"):
            lower, upper = val["range"]
            assert type(lower) == type(upper), "lower and upper values must be of same type"
            dtype = type(lower)
        else: # the other option should be "choice" but that has not been implemented
            raise f"{val.keys()} not implemented as an option yet"
        
        if hp:
            val = clip(dtype(hp.pop(0)), lower, upper)
        else:
            val = sample_value(lower, upper, dtype=dtype)

        stg_hp.append(val)
        stg_bounds.append([lower, upper])    
    
    # for stg_idx in range(stage_idx):
    #     stage_params = hparams[stg_idx]
    #     stage_params_filtered = {}
    #     for sub_stage, sub_hparams in stage_params.items():
    #         main_params = dict(filter(lambda k: not k[0].startswith("__"), sub_hparams.items()))
    #         stage_params_filtered[sub_stage] = main_params
    #     hparams[stg_idx] = stage_params_filtered
        
    # hparams[stage_idx] = hparams_out
    # hp_out_path = Path("/home/ridwan/workdir/cost_aware_bo/inputs") / hp_path.name
    # write_hparams(hparams, hp_out_path)
    return stg_hp, stg_bounds, hp

def generate_hparams(hp:List=None, hp_stgs:List=None):
    '''When a new set of hyperparaters, hp, is provided from the bayesian optimizer, this function saves the hyperparameter values for
    each stage in the respective files. When hp is none (i.e. when generating the warm up values before running BO), the function for 
    generating the hps for each stage will randomly sample from a range of values
    
    hp_stgs: list of paths where stage hyperparameters are stored. It must be arranged in the run right order
    '''
    new_x, bounds, model_input = [], [], []
    hp = hp[:] if hp else hp
    for stg_idx, hp_stg in enumerate(hp_stgs):
        out_hp_stg, stg_bounds, hp = generate_update_stage_hparams(hp, hp_stg, stg_idx)
        new_x.extend(out_hp_stg)
        bounds.append(stg_bounds)
        model_input.append(out_hp_stg)
        
    return new_x, list(zip(*bounds)), model_input
    
def read_stg_costs(stage_costs_outputs):
    new_stg_costs  = [] 
    stg_json = {}
    for i in range(len(stage_costs_outputs)):
        if not os.path.exists(stage_costs_outputs[i]):
            print(f"Cannot find {stage_costs_outputs[i]}")
            if i>1:
                raise FileNotFoundError(f"Cannot find {stage_costs_outputs[i]}")
            break 
        with open(stage_costs_outputs[i]) as stg:
            stg = stg.read().strip()
            print("param_gen 125", stg, stage_costs_outputs[i], end="\n\n")
            stg_json = json.loads(stg[:stg.find("}")+1] if "}" in stg else f"{stg}{'}'}")
            new_stg_costs.append(stg_json["COST"])
            
    return new_stg_costs

def read_objective(obj_output):
    if not Path(obj_output).exists():
        print(obj_output)
        raise FileNotFoundError(f"File {obj_output} not found.")
    with open(obj_output) as f:
        obj = f.read()
        print(obj, obj_output)
        return json.loads(obj).get("OBJ",float("inf"))

def read_hp_dataset(HP_DATASET):
    if not os.path.exists(HP_DATASET):
        return {"x":[], "y":[], "c":[]}
    with open(HP_DATASET) as f:
        return json.load(f)

def generate_hps(
    iteration,
    trial_number=1, 
    acq_type="EEIPU", 
    stage_costs_outputs:Dict[int, str]={
        0:"./stage_zero_cost.json", 1:"./stage_one_cost.json", 2:"./stage_two_cost.json",
    },
    obj_output: str = "./score.json",
    hp_stgs = [], # The number of paths provided should match the number of stages
    # config_file:Path = Path("")
):
    tic = time.time()
    bo_params = read_json(BASE_DIR/TUUN_DIR/"params-melanoma")
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
    new_x, x_bounds, model_input = generate_hparams(new_x, hp_stgs)
    
    ## Execute model (PIRLib)
    
    # resp = subprocess.run(["python", "stacking_algo.py"], capture_output=True, text=True) # For testing
    keys = stage_costs_outputs.keys()
    cmd = ["python3", "./melanoma/melanoma.py", "--hps", *[str(i) for i in new_x], "--metrics-path", str(METRICS_PATH)]
    print("Executing...", " ".join(cmd))
    cmd.append("--disable-cache") if acq_type=="EI" else None
    resp = subprocess.run(cmd, )
    
    # print(resp.stdout)
    print("Done running model")
    # print(resp.stderr)
    while not Path(obj_output).exists():
        raise FileNotFoundError(obj_output)
    
    
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

if __name__ == "__main__":
    t = time.localtime()
    parser = ArgumentParser()
    parser.add_argument("--trial", type=int, help="The trial number", default=0)
    # parser.add_argument("--iter", type=int, help="The iteration count, starting from 0. Pass -1 to only read the final objective and cost values.", required=True)
    # parser.add_argument("--stage_costs_outputs", nargs='+', help="The output directories of the stages with tunable hps in order.", required=True)
    # parser.add_argument("--obj-output", help="The output directory of the objective.", required=True)
    parser.add_argument("--base-dir", type=Path, help="Base tuun directory.", default=Path("."))
    parser.add_argument("--exp-name", type=str, help="Experiment name.")
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=0.95)
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPU", "EIPU-MEMO", "EI", "RAND"], default="EEIPU")

    args = parser.parse_args()
    
    TUUN_DIR = "cost-aware-bo"
        
    BASE_DIR = args.base_dir
    EXP_NAME = args.exp_name
    METRICS_PATH = BASE_DIR / TUUN_DIR / "metrics" / EXP_NAME
    HP_DATASET = METRICS_PATH / "hp_dataset.json"
        
    stages = ["stage1_cost.json", "stage2_cost.json"]
    stage_costs_outputs = {ind:METRICS_PATH/file for ind,file in enumerate(stages)}
    
    obj_output = METRICS_PATH / "obj.json"
    
    hp_stgs = [
        dict(
            bright_limit=dict(range=[0.1, 0.8]), 
            cont_limit=dict(range=[0.1, 0.8]), 
            p=dict(range=[0.1, 0.8])), 
        dict(
            init_lr=dict(range=[1e-6, 1e-4]),
            warmup_epo=dict(range=[1, 3]),
            h1=dict(range=[128, 1024]),
            h2=dict(range=[32, 512]),
            drp_out_p=dict(range=[0.1, 0.8]))
        ]
    params = read_json(BASE_DIR / TUUN_DIR / "params-melanoma")
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
            hp_stgs=hp_stgs,
            acq_type=args.acqf,
        )        
    print("Done!!!")
    