from cost_aware_bo.functions.processing_funcs import get_initial_data
from .MSET import MSET, Node
import numpy as np
import random
import torch
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_partitions(input_bounds, h_ind, n_stages):
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

def get_pdf(n_leaves):
    unif_prob = 1.0/n_leaves
    probs = np.array([unif_prob for i in range(n_leaves)])
    return probs

def build_tree(partitions, depths, last_stage_partition):
    root = Node(None, 0, 0)
    mset = MSET(partitions, depths, last_stage_partition)
    
    left = mset.ConstructMSET(root, 0, 0, 1, [], [[], []])
    right = mset.ConstructMSET(root, 0, 1, 2, [], [[], []])
    root.add_child(left, right)

    mset.assign_leaf_ranges(root)

    return mset, root

def get_subtree_arms(root, prev_h, prev_arm_idx):
    
    node = copy.deepcopy(root)
    curr_depth = 0
    
    while curr_depth < prev_h:
        if node.left.leaf_ranges[0] <= prev_arm_idx <= node.left.leaf_ranges[1]:
            node = node.left
        elif node.right.leaf_ranges[0] <= prev_arm_idx <= node.right.leaf_ranges[1]:
            node = node.right
            
        curr_depth += 1

    return node.leaf_ranges

def select_arm(root, leaves, probs, prev_h, prev_arm_idx, n_leaves):
    

    arm_choices = np.array([i for i in range(n_leaves)])
    valid_arm_idx = get_subtree_arms(root, prev_h, prev_arm_idx)

    valid_arm_choices = arm_choices[valid_arm_idx[0]:valid_arm_idx[1]+1]
    valid_probs = probs[valid_arm_idx[0]:valid_arm_idx[1]+1]
    
    arm_idx = random.choices(valid_arm_choices, weights=valid_probs)[0]

    return torch.tensor(leaves[arm_idx], device=DEVICE), arm_idx

def update_loss_estimators(loss, root, probs, arm_idx, sigma, H, acq_value, eta=1):
    
    loss[arm_idx][0] = acq_value
    node = copy.deepcopy(root)
    
    for height in range(1, H):

        # Move to the child that has the current arm as a leaf
        if node.left.leaf_ranges[0] <= arm_idx <= node.left.leaf_ranges[1]:
            node = node.left
        elif node.right.leaf_ranges[0] <= arm_idx <= node.right.leaf_ranges[1]:
            node = node.right
    
        nominator = 0
        for leaf_idx in range(node.leaf_ranges[0], node.leaf_ranges[1]):
            nominator += (probs[leaf_idx] * np.exp(-eta * (1 + sigma[height-1]) * loss[leaf_idx][height-1]))
    
        denominator = probs[node.leaf_ranges[0]:node.leaf_ranges[1]+1].sum()

        # print(f'To update losses, the nominator is {nominator} and the denominator is {denominator}')
        
        loss_i = np.log( nominator / denominator )**(-1/eta)

        loss[arm_idx][height] = sigma[height] * loss_i

    return loss

def update_arm_probability(loss, probs, arm_idx, n_leaves, eta=1):
        
    nominator = probs[arm_idx] * np.exp(-eta*loss[arm_idx,:].sum())
    
    denominator = 0
    for leaf_idx in range(n_leaves):
        denominator += probs[leaf_idx] * np.exp(-eta*loss[leaf_idx,:].sum())

    return nominator/denominator

def update_all_probabilities(loss, probs, arm_idx, n_leaves):

    # We update the probability of the current arm first
    probs[arm_idx] = update_arm_probability(loss, probs, arm_idx, n_leaves)

    # We then update the probabilities of all remaining arms
    for idx in range(n_leaves):
        if idx == arm_idx:
            continue
        probs[idx] = update_arm_probability(loss, probs, idx, n_leaves)

    return probs

def remove_invalid_partitions(input_bounds, probs, loss, h_ind, n_leaves, H, n_stages, leaf_partitions):
    prob_thres = 1e-6
                                                        
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

        probs = get_pdf(n_leaves)
        loss = np.zeros([n_leaves, H])

    return input_bounds, probs, loss

# def build_datasets(self, acqf, leaf_bounds, trial_number, n_leaves, arm_idx, params):

#     X_tree, Y_tree, C_tree, C_inv_tree = [], [], [], []
#     best_f = -1e9
#     init_cost = 0
#     for leaf in range(n_leaves):
#         x, y, c, c_inv, cost0 = get_initial_data(
#             params['n_init_data'], bounds=leaf_bounds[leaf], 
#             seed=trial_number*10000, acqf=acqf, params=params)
        
#         X_tree.append(x)
#         Y_tree.append(y)
#         C_tree.append(c)
#         C_inv_tree.append(c_inv)

#         if leaf == arm_idx:
#             init_cost = cost0
#             best_f = max(best_f, y.max().item())

#     return X_tree, Y_tree, C_tree, C_inv_tree, init_cost, best_f

def build_datasets(acqf, bounds, trial_number, n_leaves, arm_idx, params):

    X, Y, C, C_inv, cost0 = get_initial_data(
        params['n_init_data'], bounds=bounds, 
        seed=trial_number*10000, acqf=acqf, params=params)
    
    best_f = Y.max().item()

    return X, Y, C, C_inv, cost0, best_f