from optimizer.optimize_acqf_funcs import optimize_acqf
from itertools import chain
import torch
import copy

def update_candidate(candidate, acqf_val, best_candidate, best_acqf_val, num_memoised, delta):
    if acqf_val > best_acqf_val:
        return candidate.detach(), acqf_val, delta
    return best_candidate, best_acqf_val, num_memoised

def optimize_acqf_by_mem(acqf=None, acqf_str=None, bounds=None, iter=None, params=None, prefix_pool=[[]], seed=0):
    n_memoised = 0
    
    best_candidate, best_acqf_val = -torch.inf, -torch.inf
    for prefix in prefix_pool:
        cand_generation_bounds = copy.deepcopy(bounds)
        pref_stages = len(prefix)
        prefix = list(chain(*prefix))
        
        for i, pref_param in enumerate(prefix):
            cand_generation_bounds[0][i], cand_generation_bounds[1][i] = pref_param, pref_param
        
        new_candidate, acqf_val = optimize_acqf(
            acq_function=acqf, acq_type=acqf_str, delta=pref_stages, 
            curr_iter=iter, bounds=cand_generation_bounds, q=1, num_restarts=10, 
            raw_samples=512, options={'seed': seed})
            
        best_candidate, best_acqf_val, n_memoised = update_candidate(
            new_candidate, acqf_val.item(), best_candidate, 
            best_acqf_val, n_memoised, pref_stages)
    
    return best_candidate, n_memoised, best_acqf_val
        
