import yaml
from collections import OrderedDict
import json
from typing import List, Dict, Union
from pathlib import Path
import sys
sys.path.append("../codeformer/")

import numpy as np

from basicsr.utils.options import parse


# TODO: Define bounds for hyperparameter values that are bounded
def sample_value(
    lower, upper, dtype=float, choice_list=[]
):
    assert not(dtype and choice_list), "Only one of dtype and choice_list should be set to a value"
    if dtype==float:
        val = np.random.random()
        val = val * (upper - lower) + lower
    elif dtype==int:
        val = np.random.randint(lower, upper)
    elif dtype==str:
        val = choice_list[np.random.randint(0, len(choice_list))]
    else:
        raise ValueError("Invalid `dtype` provided. Provide one of `int` and `float`")
    
    return val
    
def clip(data:Union[List, int, float], lower:Union[int, float]=None, upper:Union[int, float]=None):
    if not lower and not upper:
        return data
    if isinstance(data, list):
        return [clip(val, lower, upper) for val in data]
    return max(lower, min(upper, data))


def generate_update_stage_hparams(hp: List, hp_path):
    hp_path = Path(hp_path)
    with hp_path.open("r") as f:
        opt: OrderedDict = parse(hp_path, "../codeformer")
        
    optimizable: OrderedDict = opt.pop("optimizable")
    hparams: OrderedDict = opt["train"]
    
    stg_hp, stg_bounds = [], []
    hparams_out = {}
    for key1, value1 in optimizable.items():
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                lower, upper = value2
        else:
            lower, upper = value2
        
        dtype = type(lower)
        if hp:
            val = clip(dtype(hp.pop(0)), lower, upper)
        else:
            val = sample_value(lower, upper, dtype=dtype)
        
        stg_hp.append(val)
        stg_bounds.append([lower, upper])
        if isinstance(value1, dict):
            hparams[key1][key2] = val
        else:
            hparams[key1] = val
                
    opt["train"] = hparams
    
    hp_out_path = hp_path.parent / hp_path.name[hp_path.name.find("stage"):]
    opt_dict = json.loads(json.dumps(opt))
    print(opt_dict)
    with hp_out_path.open("w") as f:
        yaml.dump(opt_dict, f, sort_keys=False)
    # return stg_hp, stg_bounds, hp

if __name__=="__main__":
    generate_update_stage_hparams([], "../codeformer/options/CodeFormer_stage3.yml")