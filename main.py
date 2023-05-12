import wandb
from argparse import ArgumentParser
import time
from copy import deepcopy

import botorch
from optimizer.optimize_acqf_funcs import optimize_acqf, _optimize_acqf_batch, gen_candidates_scipy, gen_batch_initial_conditions
from json_reader import read_json
from single_trial import bo_trial
import torch
import random
import numpy as np

def arguments():
    parser = ArgumentParser()
    parser.add_argument("--obj-funcs", nargs="+", help="Objective functions", default=["beale2", "hartmann3", "beale2"])
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--cost-types", nargs="+", help="Cost types", default=[1,2,3])
    parser.add_argument("--warmup-eta", type=float, help="Warm up", default=1e-2)
    parser.add_argument("--trial-num", type=int, help="Trial number")
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPU", "EIPU-MEMO", "EI", "RAND"])
    
    params:dict = read_json("params")
    
    args = parser.parse_args()
    
    args_dict = deepcopy(vars(args))
    args_dict.pop("trial_num")
    args_dict.pop("exp_group")
    params.update(args_dict)
    
    return args, params

if __name__=="__main__":
    args, params = arguments()
    trial = args.trial_num
    wandb.init(
        entity="cost-bo",
        project="memoised-cost-aware-bo-organized",
        group=f"{args.exp_group}--acqf_{args.acqf}|-obj-func_{'-'.join(args.obj_funcs)}|-dec-fac_{args.decay_factor}"
                f"|init-eta_{args.init_eta}|-cost-typ_{'-'.join(list(map(str,args.cost_types)))}",
        name=f"{time.strftime('%Y-%m-%d-%H%M')}-trial-number_{trial}",
        config=params
    )
    botorch.optim.optimize.optimize_acqf = optimize_acqf
    botorch.optim.optimize._optimize_acqf_batch = _optimize_acqf_batch
    botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy
    botorch.optim.initializers.gen_batch_initial_conditions = gen_batch_initial_conditions
    
    
    torch.manual_seed(seed=params['rand_seed'])
    np.random.seed(params['rand_seed'])
    random.seed(params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=params['rand_seed'])
    
    logs = read_json('logs')
    
    # for trial in range(1, params['n_trials'] + 1):
    trial = args.trial_num
    bo_trial(trial_number=trial, acqf=args.acqf, wandb=wandb, params=params)
    
    # for key in logs.keys():
    #     logs[key].append(trial_logs[key])
    #     if trial == params['n_trials']:
    #         logs[key] = np.array(logs[key])
    
    
