from json_reader import read_json
from functions import generate_input_data, F, Cost_F, get_gen_bounds, get_dataset_bounds
from single_iteration import bo_iteration
import numpy as np
import torch
from typing import Iterable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_initial_data(n, bounds=None, seed=0, params=None):
    X = generate_input_data(N=n, bounds=bounds, seed=seed)
    y = F(X, params).unsqueeze(-1)
    c = Cost_F(X, params)

    return X, y, c

def print_iter_logs(trial_logs):

    print(f"Best F = {trial_logs['best f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"Predicted Y = {trial_logs['f^(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"f(x) = {trial_logs['f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"c(x) = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['c(x)'][-1]) + "]", end = '\t\t')
    print(f"E[c(x)] = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['E[c(x)]'][-1]) + "]", end = '\t\t')
    print(f"sum(c(x)) = [{trial_logs['sum(c(x))'][-1]:>4.3f}]", end = '\t\t')
    print(f"sum(E[c(x)]) = {trial_logs['sum(E[c(x)])'][-1]:>4.3f}", end = '\t\t')
    print(f"Cum Costs = {trial_logs['cum(costs)'][-1]:>4.3f}", end = '\t\t')
    print(f"Inverse c(x) = {trial_logs['1/c(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"E[inv_c(x)] = {trial_logs['E[1/c(x)]'][-1]:>4.3f}")
    print('\n')

def bo_trial(trial_number, acqf, wandb, params=None):

    trial_logs = read_json('logs')
    bound_list = read_json('bounds')
    
    chosen_functions, h_ind = params['obj_funcs'], params['h_ind']
    
    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
    X, Y, C = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number, params=params)
    
    trial_logs['cum(costs)'].append(sum([stage_costs.sum() for stage_costs in C]).item())
    trial_logs['f^(x)'].append(Y.max().item())
    
    eta = 1.0
    for iteration in range(params['n_iters']):
        bounds = get_dataset_bounds(X, Y, C, input_bounds)
        
        new_x, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(X, Y, C, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, params=params)
        
        if iteration > params['warmup_iters']:
            eta *= params['decay_factor']
        
        new_y = F(new_x, params).unsqueeze(-1)
        
        new_c = Cost_F(new_x, params)
        inv_cost = 1/sum(new_c)
        
        X = torch.cat([X, new_x])
        Y = torch.cat([Y, new_y])
        for stage, _ in enumerate(C):
            C[stage] = torch.cat([C[stage], new_c[stage]])
            
        best_f = Y.max().item()
        stage_cost_list = [stage.item() for stage in new_c]
        sum_stages = sum(stage_cost_list)
        sum_Ec = sum(E_c)
        
        trial_logs['best f(x)'].append(best_f)
        trial_logs['x'] = np.array(new_x.cpu())
        trial_logs['f^(x)'].append(y_pred)
        trial_logs['f(x)'].append(new_y.item())
        trial_logs['c(x)'].append(stage_cost_list)
        trial_logs['sum(c(x))'].append(sum_stages)
        trial_logs['cum(costs)'].append(trial_logs['cum(costs)'][-1] + sum_stages)
        trial_logs['E[c(x)]'].append(E_c)
        trial_logs['E[1/c(x)]'].append(E_inv_c.item())
        trial_logs['1/c(x)'].append(inv_cost.item())
        trial_logs['sum(E[c(x)])'].append(sum_Ec)
        
        if params['verbose']:
            print(f"Iteration-{iteration} [{acqf}] Trial #{trial_number} Current X = {np.array(new_x.cpu())[0]}")
            print_iter_logs(trial_logs)
        
        log = dict(
            best_f=best_f,
            f_hat_x=y_pred,
            f_x=new_y.item(),
            sum_c_x=sum_stages,
            cum_costs=trial_logs['cum(costs)'][-1],
            E_inv_c=E_inv_c,
            sum_Ec=sum_Ec,
            inv_cost=inv_cost,
            E_c=dict(zip(map(str,range(len(E_c))) ,E_c)),
            c_x=dict(zip(map(str,range(len(stage_cost_list))) ,stage_cost_list)),
            c_res=dict(zip(map(str,range(len(stage_cost_list))) ,[abs(act-est) for act, est in zip(E_c,stage_cost_list)])),
            inv_c_res=abs(E_inv_c.item()-inv_cost.item())
        )
        # log.update(dict(zip(map(str,range(len(E_c))) ,E_c)))
        wandb.log(log)
        
    return trial_logs


        
#        print('Costs = ')
#        print(C)
#
#        print('X bounds = ')
#        print(bounds['x'])
#
#        print('X norm bounds = ')
#        print(bounds['x_cube'])
#
#        print('Y bounds = ')
#        print(bounds['y'])
#
#        print('Cost bounds = ')
#        print(bounds['c'])
