import torch
import numpy as np
import random
import botorch
from single_trial import bo_trial
from json_reader import read_json
    
def run_trials(acqf='', wandb=print, params=None):
    
    torch.manual_seed(seed=params['rand_seed'])
    np.random.seed(params['rand_seed'])
    random.seed(params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=params['rand_seed'])
    
    logs = read_json('logs')
    
    for trial in range(1, params['n_trials'] + 1):
        trial_logs = bo_trial(trial_number=trial, acqf=acqf, wandb=wandb)
        
        for key in logs.keys():
            logs[key].append(trial_logs[key])
            if trial == params['n_trials']:
                logs[key] = np.array(logs[key])
    return logs

