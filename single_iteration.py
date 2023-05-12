from optimize_mem_acqf import optimize_acqf_by_mem
from EEIPU.EIPUVariants import EIPUVariants
from functions import normalize, unnormalize, normalize_cost, standardize, unstandardize, initialize_GP_model, get_gen_bounds, generate_prefix_pool, Cost_F
from json_reader import read_json
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
import torch

def get_gp_models(X, y, params=None):
    mll, gp_model = initialize_GP_model(X, y, params=params)
    fit_gpytorch_model(mll)
    return mll, gp_model

def get_cost_models(X, C, param_idx, bounds):
    cost_mll, cost_gp = [], []
    for i, stage_cost in enumerate(C):
        log_sc = torch.log(stage_cost)
        norm_stage_cost = standardize(log_sc, bounds['c'][:,i])
        
        stage_idx = param_idx[i]
        stage_c = X[:, stage_idx] + 0
        
        stage_mll, stage_gp = get_gp_models(stage_c, norm_stage_cost)
        
        cost_mll.append(stage_mll)
        cost_gp.append(stage_gp)
    return cost_mll, cost_gp
    
def get_expected_y(X, gp_model, n_samples, bounds, seed):
    sampler = SobolQMCNormalSampler(sample_shape=n_samples, seed=seed)
    acq_obj = IdentityMCObjective()
    posterior = gp_model.posterior(X)
    
    samples = sampler(posterior)
    samples = samples.max(dim=2)[0]
    samples = unnormalize(samples, bounds=bounds)
    
    obj = acq_obj(samples)
    obj = obj.mean(dim=0).item()
    
    return obj

def bo_iteration(X, y, c, bounds=None, acqf_str='', decay=None, iter=None, params=None):
    
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    mll, gp_model = get_gp_models(train_x, train_y, params=params)
    
    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    prefix_pool = None
    if params['use_pref_pool']:
        prefix_pool = generate_prefix_pool(train_x, acqf_str, params)
    
    if acqf_str == 'RAND':
        acqf=acqf_str
        norm_bounds = bounds['x'] + 0
    elif acqf_str == 'EI':
        acqf = ExpectedImprovement(model=gp_model, best_f=train_y.max())
    elif 'EIPU' in acqf_str:
        cost_mll, cost_gp = None, None
        if acqf_str == 'EEIPU':
            cost_mll, cost_gp = get_cost_models(train_x, c, params['h_ind'], bounds)
        
        cost_sampler = SobolQMCNormalSampler(sample_shape=params['cost_samples'], seed=params['rand_seed'])
        acqf = EIPUVariants(acq_type=acqf_str, model=gp_model, cost_gp=cost_gp, best_f=train_y.max(), cost_sampler=cost_sampler,
            acq_objective=IdentityMCObjective(), cost_func=Cost_F, unstandardizer=unstandardize,
            unnormalizer=unnormalize, cost_normalizer=normalize_cost, bounds=bounds, eta=decay, params=params)
    
    new_x, n_memoised = optimize_acqf_by_mem(acqf=acqf, acqf_str=acqf_str, bounds=norm_bounds, iter=iter, prefix_pool=prefix_pool, seed=params['rand_seed'])
    
    E_c, E_inv_c = [0], torch.tensor([0])
    if acqf_str == 'EEIPU':
        E_c = acqf.compute_expected_cost(new_x)
        E_inv_c = acqf.compute_expected_inverse_cost(new_x[:, None, :])
    E_y = get_expected_y(new_x, gp_model, params['cost_samples'], bounds['x_cube'], params['rand_seed'])

    if acqf_str != 'RAND':
        new_x = unnormalize(new_x, bounds=bounds['x_cube'])
    
    
    return new_x, n_memoised, E_c, E_inv_c, E_y

