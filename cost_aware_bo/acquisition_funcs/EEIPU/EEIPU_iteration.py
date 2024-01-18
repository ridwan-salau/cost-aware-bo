from acquisition_funcs.EEIPU.EEIPU import EEIPU
import torch
from functions.processing_funcs import normalize, unnormalize, standardize, unstandardize, get_gen_bounds, generate_prefix_pool
from functions.iteration_funcs import get_gp_models, get_multistage_cost_models
from optimize_mem_acqf import optimize_acqf_by_mem
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective

def eeipu_iteration(X, y, c, bounds=None, acqf_str='', decay=None, iter=None, consumed_budget=None, params=None):

    # Next test: try bounds['x'] again and make sure prefix pooling works with normalized data
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    prefix_pool = None
    if params['use_pref_pool']:
        prefix_pool = generate_prefix_pool(train_x, y, acqf_str, params)

    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    
    cost_mll, cost_gp = get_multistage_cost_models(train_x, c, iter, params['h_ind'], bounds, acqf_str)
        
    cost_sampler = SobolQMCNormalSampler(sample_shape=params['cost_samples'], seed=iter)
    acqf = EEIPU(acq_type=acqf_str, model=gp_model, cost_gp=cost_gp, best_f=train_y.max(),
                 cost_sampler=cost_sampler, acq_objective=IdentityMCObjective(),
                 unstandardizer=unstandardize, normalizer = normalize, unnormalizer=unnormalize, 
                 bounds=bounds, eta=decay, consumed_budget=consumed_budget, iter=iter, params=params)
    
    new_x, n_memoised, acq_value = optimize_acqf_by_mem(
        acqf=acqf, acqf_str=acqf_str, bounds=norm_bounds, 
        iter=iter, prefix_pool=prefix_pool, params=params, seed=iter)

    new_x = unnormalize(new_x, bounds['x_cube'])
    
    return new_x, n_memoised, acq_value