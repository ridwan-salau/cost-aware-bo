import copy
import numpy as np
import torch
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node:
    def __init__(self, parent, depth, idx):
        self.parent = parent
        self.depth = depth
        self.idx = idx
        self.left = None
        self.right = None
        self.value = None
        self.leaf_partitions = None
        self.leaf_ranges = None

    def add_parent(self, parent):
        self.parent = parent

    def add_child(self, left, right):
        self.left = left
        self.right = right

    def add_value(self, value):
        self.value = value

    def add_leaf_partition(self, leaf_partitions):
        self.leaf_partitions = leaf_partitions

    def get_depth(self):
        return self.depth

    def get_idx(self):
        return self.idx

class MSET:
    def __init__(self, partitions, depths, last_stage_bounds):
        self.partitions = partitions
        self.depths = depths
        self.stage = 0
        self.last_stage_bounds = last_stage_bounds
        self.leaves = []
        self.leaf_partitions = []

    def update_bounds(self, curr_stage_partitions, p_bounds, p_idx):
        bounds = copy.deepcopy(p_bounds)
        curr_partition = []

        for hp_partitions in curr_stage_partitions:
            curr_partition.append(hp_partitions[p_idx])

        for ps in curr_partition:
            bounds[0].append(ps[0].item())
            bounds[1].append(ps[1].item())

        return bounds

    def create_leaf(self, node, leaf_partition, bounds):
        for ps in self.last_stage_bounds:
            bounds[0].append(ps[0].item())
            bounds[1].append(ps[1].item())
        # bounds = torch.tensor(bounds, device=DEVICE, dtype=torch.double)
        node.add_value(bounds)
        node.add_leaf_partition(leaf_partition)
        
        self.leaves.append(bounds)
        self.leaf_partitions.append(leaf_partition)
        
        return node

    def build_children(self, node, stage_idx, node_idx, leaf_partition, bounds):
        
        left_range, right_range = 1e9, -1e9
        
        left = self.ConstructMSET(node, stage_idx + 1, 0, 2*node_idx + 1, leaf_partition, bounds)
        right = self.ConstructMSET(node, stage_idx + 1, 1, 2*node_idx + 2, leaf_partition, bounds)
        node.add_child(left, right)

        return node

    def ConstructMSET(self, parent, stage_idx, p_idx, node_idx, leaf_partitions=None, p_bounds=None):

        leaf_partition = copy.deepcopy(leaf_partitions)

        curr_stage_partitions = self.partitions[stage_idx]

        leaf_partition.append(p_idx)
        
        bounds = self.update_bounds(curr_stage_partitions, p_bounds, p_idx)
        
        curr_depth = parent.get_depth() + self.depths[stage_idx]

        node = Node(parent, curr_depth, node_idx)

        if stage_idx >= len(self.depths) - 1:
            node = self.create_leaf(node, leaf_partition, bounds)
            return node

        node = self.build_children(node, stage_idx, node_idx, leaf_partition, bounds)

        return node

    def assign_leaf_ranges(self, node):
        # Base case: If the node is a leaf, its range is just its index
        if not node.left and not node.right:
            first_idx = 2**node.depth - 1
            node.leaf_ranges = (node.idx - first_idx, node.idx - first_idx)
            return node.leaf_ranges
    
        # Recursive case: Compute ranges for left and right subtrees
        left_range = self.assign_leaf_ranges(node.left) if node.left else None
        right_range = self.assign_leaf_ranges(node.right) if node.right else None
    
        # Combine the ranges from left and right children
        start = left_range[0] if left_range else right_range[0]
        end = right_range[1] if right_range else left_range[1]
        node.leaf_ranges = (start, end)
    
        return node.leaf_ranges

    def get_leaves(self):
        return self.leaves

    def get_leaf_partitions(self):
        return self.leaf_partitions

    def print_MSET(self, node):
        
        if node.left is not None:
            # print(node.idx, ' ', node.leaf_ranges)
            self.print_MSET(node.left)
            self.print_MSET(node.right)
        else:
            print(node.idx)
            print(node.value)
            print(node.leaf_partitions)
            print(node.leaf_ranges)
            print('\n\n')