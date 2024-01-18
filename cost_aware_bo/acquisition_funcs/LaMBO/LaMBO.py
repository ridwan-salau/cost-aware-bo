from acquisition_funcs.LaMBO.LaMBO_iteration import lambo_iteration
from json_reader import read_json
from functions.iteration_funcs import iteration_logs
from functions.processing_funcs import get_gen_bounds, get_dataset_bounds, get_initial_data
from functions.synthetic_functions import Cost_F, F
from acquisition_funcs.LaMBO.MSET import MSET, Node
import numpy as np
import random
import torch
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LaMBO:
    def __init__(self, eta):
        self.eta = eta
        return

    def build_partitions(self, input_bounds, h_ind, n_stages):
        partitions = []
        for i in range(n_stages-1):
            stage_partition = []
            for stage_idx in h_ind[i]:
                lo, hi = input_bounds[0][stage_idx], input_bounds[1][stage_idx]
                mid = (lo + hi) / 2.0
                p = [[lo, mid], [mid, hi]]
                stage_partition.append(p)
            partitions.append(stage_partition)
    
        last_stage_partition = []
        for stage_idx in h_ind[-1]:
            lo, hi = input_bounds[0][stage_idx], input_bounds[1][stage_idx]
            p = [lo, hi]
            last_stage_partition.append(p)
    
        return partitions, last_stage_partition
    
    def get_pdf(self, n_leaves):
        unif_prob = 1.0/n_leaves
        probs = np.array([unif_prob for i in range(n_leaves)])
        return probs
    
    def build_tree(self, partitions, depths, last_stage_partition):
        root = Node(None, 0, 0)
        mset = MSET(partitions, depths, last_stage_partition)
        
        left = mset.ConstructMSET(root, 0, 0, 1, [], [[], []])
        right = mset.ConstructMSET(root, 0, 1, 2, [], [[], []])
        root.add_child(left, right)
    
        mset.assign_leaf_ranges(root)
    
        return mset, root
    
    def get_subtree_arms(self, root, prev_h, prev_arm_idx):
        
        node = copy.deepcopy(root)
        curr_depth = 0
        
        while curr_depth < prev_h:
            if node.left.leaf_ranges[0] <= prev_arm_idx <= node.left.leaf_ranges[1]:
                node = node.left
            elif node.right.leaf_ranges[0] <= prev_arm_idx <= node.right.leaf_ranges[1]:
                node = node.right
                
            curr_depth += 1
    
        return node.leaf_ranges
    
    def select_arm(self, root, leaves, probs, prev_h, prev_arm_idx, n_leaves):
        
    
        arm_choices = np.array([i for i in range(n_leaves)])
        valid_arm_idx = self.get_subtree_arms(root, prev_h, prev_arm_idx)
    
        valid_arm_choices = arm_choices[valid_arm_idx[0]:valid_arm_idx[1]+1]
        valid_probs = probs[valid_arm_idx[0]:valid_arm_idx[1]+1]
        
        arm_idx = random.choices(valid_arm_choices, weights=valid_probs)[0]
    
        return torch.tensor(leaves[arm_idx], device=DEVICE), arm_idx
    
    def update_loss_estimators(self, loss, root, probs, arm_idx, sigma, H, acq_value):
        loss[arm_idx][0] = acq_value
        node = copy.deepcopy(root)
        for height in range(1, H):
        
            if node.left.leaf_ranges[0] <= arm_idx <= node.left.leaf_ranges[1]:
                node = node.left
            elif node.right.leaf_ranges[0] <= arm_idx <= node.right.leaf_ranges[1]:
                node = node.right
        
            nominator = 0
            for leaf_idx in range(node.leaf_ranges[0], node.leaf_ranges[1]):
                nominator += (probs[leaf_idx] * np.exp(-self.eta * (1 + sigma[height-1]) * loss[leaf_idx][height-1]))
        
            denominator = probs[node.leaf_ranges[0]:node.leaf_ranges[1]+1].sum()

            print(f'To update losses, the nominator is {nominator} and the denominator is {denominator}')

            eps = 1e-5
            if denominator <= eps:
                denominator = 1
            
            loss_i = np.log( nominator / denominator )**(-1/self.eta)
    
            loss[arm_idx][height] = sigma[height] * loss_i
    
        return loss
    
    def update_arm_probability(self, loss, probs, arm_idx, n_leaves):
        
        nominator = probs[arm_idx] * np.exp(-self.eta*loss[arm_idx,:].sum())
        
        denominator = 0
        for leaf_idx in range(n_leaves):
            denominator += probs[leaf_idx] * np.exp(-self.eta*loss[leaf_idx,:].sum())

        print(f'To update probabilities, the nominator is {nominator} and the denominator is {denominator}')
        eps = 1e-5
        if denominator <= eps:
            denominator = 1
    
        probs[arm_idx] = nominator/denominator
    
        return probs
    
    def remove_invalid_partitions(self, input_bounds, probs, h_ind, n_leaves, n_stages, leaf_partitions):
        prob_thres = 0.1/n_leaves
                                                            
        invalid_partitions = np.where(probs < prob_thres)[0]
        
        if invalid_partitions.shape[0] > 0:
            first_invalid_idx = invalid_partitions[0]
            invalid_partition = leaf_partitions[first_invalid_idx]
            
            for i in range(n_stages-1):
                for stage_idx in h_ind[i]:
                    if invalid_partition[i] == 0:
                        input_bounds[0][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0
                    else:
                        input_bounds[1][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0
    
        return input_bounds
    
    def build_datasets(self, acqf, leaf_bounds, trial_number, n_leaves, params):
    
        X_tree, Y_tree, C_tree, C_inv_tree = [], [], [], []
        best_f = -1e9
        init_cost = 0
        for leaf in range(n_leaves):
            x, y, c, c_inv, cost0 = get_initial_data(
                params['n_init_data'], bounds=leaf_bounds[leaf], 
                seed=trial_number*10000, acqf=acqf, params=params)
            
            X_tree.append(x)
            Y_tree.append(y)
            C_tree.append(c)
            C_inv_tree.append(c_inv)
            
            init_cost += cost0
            best_f = max(best_f, y.max().item())
            
        return X_tree, Y_tree, C_tree, C_inv_tree, init_cost, best_f
    
    def lambo_trial(self, trial_number, acqf, wandb, params=None):
        
        chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']
        bound_list = read_json('bounds')

        global_input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
        n_stages = len(h_ind)
        n_leaves = 2**(n_stages-1)
        
        probs = self.get_pdf(n_leaves)

        partitions, last_stage_partition = self.build_partitions(global_input_bounds, h_ind, n_stages)
    
        depths = [ 1 for i in range(n_stages - 1) ]
        
        mset, root = self.build_tree(partitions, depths, last_stage_partition)

        X_tree, Y_tree, C_tree, C_inv_tree, init_cost, best_f = self.build_datasets(acqf, mset.leaves, trial_number, n_leaves, params)
    
        H = sum(depths)
        h = H
        
        arm_idx = random.randint(0, n_leaves)
        
        print(f'Initial Data has {X_tree[arm_idx].shape} points for {acqf} Trial {trial_number} with cost {init_cost}')
        
        loss = np.zeros([n_leaves, H])
        
        best_f = -1e9
        for idx in range(arm_idx):
            best_f = max(best_f, Y_tree[idx].max().item())
            
        total_budget = params['total_budget']
        cum_cost = 500
        iteration = 0
        
        while cum_cost < total_budget:

            print(f'\n\n{loss}\n{probs}\n\n')
                
            leaf_bounds = mset.leaves
            input_bounds, arm_idx = self.select_arm(root, leaf_bounds, probs, h, arm_idx, n_leaves)
    
            X, Y, C, C_inv = X_tree[arm_idx], Y_tree[arm_idx], C_tree[arm_idx], C_inv_tree[arm_idx]
    
            bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
    
            new_x, n_memoised, acq_value = lambo_iteration(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=self.eta, iter=iteration, consumed_budget=cum_cost, params=params)
    
            sigma = np.array(random.choices([-1, 1], k=H))
            sigma[-1] = -1
    
            h = np.where(sigma == -1)[0][0]
    
            loss = self.update_loss_estimators(loss, root, probs, arm_idx, sigma, H, acq_value)
    
            probs = self.update_arm_probability(loss, probs, arm_idx, n_leaves)
    
            global_input_bounds = self.remove_invalid_partitions(input_bounds, probs, h_ind, n_leaves, n_stages, mset.leaf_partitions)
        
            partitions, last_stage_partition = self.build_partitions(global_input_bounds, h_ind, n_stages)
        
            mset, root = self.build_tree(partitions, depths, last_stage_partition)
        
            new_y = F(new_x, params).unsqueeze(-1)
            new_c = Cost_F(new_x, params)
            inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)

            if new_c.sum() > 50:
                continue
            
            new_x, new_y, new_c, inv_cost = new_x.to(DEVICE), new_y.to(DEVICE), new_c.to(DEVICE), inv_cost.to(DEVICE)
            
            X_tree[arm_idx] = torch.cat([X_tree[arm_idx], new_x])
            Y_tree[arm_idx] = torch.cat([Y_tree[arm_idx], new_y])
            C_tree[arm_idx] = torch.cat([C_tree[arm_idx], new_c])
            C_inv_tree[arm_idx] = torch.cat([C_inv_tree[arm_idx], inv_cost])
            
            best_f = max(best_f, new_y.item())
    
            sum_stages = new_c.sum().item()        
            cum_cost += sum_stages

            iteration_logs(acqf, trial_number, iteration, best_f, sum_stages, cum_cost)
            iteration += 1

    
        print(f'{acqf} Trial {trial_number} Final Data has {X_tree[arm_idx].shape} datapoints with best_f {best_f:0,.2f}')
            
            # wandb.log(log)
        