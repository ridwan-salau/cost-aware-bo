from optimizer.optimize_acqf_funcs import optimize_acqf
from functions import get_random_observations
from itertools import chain
import torch

def update_candidate(candidate, acqf_val, best_candidate, best_acqf_val, num_memoised, delta):
    if acqf_val > best_acqf_val:
        return candidate.detach(), acqf_val, delta
    return best_candidate, best_acqf_val, num_memoised

def optimize_acqf_by_mem(acqf=None, acqf_str=None, bounds=None, iter=None, prefix_pool=None, seed=0):
    n_memoised = 0
    if acqf_str == 'RAND':
        new_x = get_random_observations(N=1, bounds=bounds)
        return new_x, n_memoised
    
    best_candidate, best_acqf_val = -torch.inf, -torch.inf
    for prefix in prefix_pool:
        pref_bounds = bounds + 0
        pref_stages = len(prefix)
        prefix = list(chain(*prefix))
        
        for i, pref_param in enumerate(prefix):
            bounds[0][i], bounds[1][i] = pref_param, pref_param
        
        new_candidate, acqf_val = optimize_acqf(acq_function=acqf, acq_type=acqf_str, delta=pref_stages, curr_iter=iter, bounds=bounds, q=1, num_restarts=10, raw_samples=512, options={'seed': seed})
        
        best_candidate, best_acqf_val, n_memoised = update_candidate(new_candidate, acqf_val.item(), best_candidate, best_acqf_val, n_memoised, pref_stages)
    
    return best_candidate, n_memoised
        
