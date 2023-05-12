from json_reader import read_json
from functions import generate_input_data, F, Cost_F, get_gen_bounds, get_dataset_bounds
from single_iteration import bo_iteration
import numpy as np
from copy import deepcopy
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
    
    cum_cost = sum([stage_costs.sum() for stage_costs in C]).item()
    
    eta = params['init_eta']

    for iteration in range(params['n_iters']):
        bounds = get_dataset_bounds(X, Y, C, input_bounds)

        new_x, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(X, Y, C, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, params=params)
        
        eta *= params['decay_factor']

        new_x = new_x.to(DEVICE)
        new_y = F(new_x, params).unsqueeze(-1)
        new_y = new_y.to(DEVICE)
        new_c = Cost_F(new_x, params)
        inv_cost = 1/sum(new_c)
        
        X = torch.cat([X, new_x])
        Y = torch.cat([Y, new_y])
        for stage, _ in enumerate(C):
            C[stage] = torch.cat([C[stage], new_c[stage]])
            
        best_f = Y.max().item()
        cost_copy = deepcopy(new_c)
        cost_copy = [cost.item() for cost in cost_copy]
        for stage in range(n_memoised):
            new_c[stage] = torch.tensor([params['epsilon']])

        stage_cost_list = [stage.item() for stage in new_c]
        sum_stages = sum(stage_cost_list)
        sum_Ec = sum(E_c)
        cum_cost += sum_stages
        
        log = dict(
            best_f=best_f,
            f_hat_x=y_pred,
            f_x=new_y.item(),
            f_res=abs(y_pred - new_y.item()),
            sum_c_x=sum_stages,
            cum_costs=cum_cost,
            E_inv_c=E_inv_c,
            sum_Ec=sum_Ec,
            inv_cost=inv_cost,
            E_c=dict(zip(map(str,range(len(E_c))) ,E_c)),
            c_x=dict(zip(map(str,range(len(stage_cost_list))) ,stage_cost_list)),
            c_res=dict(zip(map(str,range(len(stage_cost_list))) ,[abs(act-est) for act, est in zip(E_c,cost_copy)])),
            inv_c_res=abs(E_inv_c.item()-inv_cost.item()),
            eta = eta
        )
        
        wandb.log(log)
        
    return


        
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
