import json
import os
import pickle
import shutil
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import torch
import wandb
from cost_aware_bo import generate_hps, log_metrics, update_dataset_new_run
import s3fs

from tuning_multi import t5_fine_tuning

s3=s3fs.S3FileSystem()

sys.path.append("./")

parser = ArgumentParser()
parser.add_argument(
    "--exp-name", type=str, required=True, help="Specifies a unique experiment name"
)
parser.add_argument("--trial", type=int, help="The trial number", default=1)
parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
parser.add_argument("--decay-factor", type=float, help="Decay factor", default=0.95)
parser.add_argument(
    "--acqf",
    type=str,
    help="Acquisition function",
    choices=["EEIPU", "MS_CArBO", "EIPS", "CArBO", "EI", "RAND"],
    default="EI",
)
parser.add_argument(
    "--cache-root", type=str, default=".cachestore", help="Cache directory"
)
parser.add_argument("--disable-cache", action="store_true", help="Disable cache")
parser.add_argument(
    "--data-dir", type=str, help="Directory with the data", default="./inputs"
)
args = parser.parse_args()
disable_cache = args.acqf != "EEIPU"

data_dir: str = args.data_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

init_dataset_path = Path(
    f"{data_dir}/{args.exp_name}/t5_init_dataset-trial_{args.trial}.pk"
)
s3.mkdirs(init_dataset_path.parent, exist_ok=True)
dataset = {}
t5_init_dataset = {}
if s3.exists(init_dataset_path):
    with s3.open(init_dataset_path, "rb") as f:
        t5_init_dataset = pickle.load(f)

with s3.open(data_dir / "initial_hparams_multi.json") as f:
    initial_hparams = json.load(f)
    hp_sampling_range = initial_hparams["hp_sampling_range"]
    params = initial_hparams["params"]

args_dict = deepcopy(vars(args))
params.update(args_dict)

date_now = f"{time.strftime('%Y-%m-%d-%H%M')}"

wandb.init(
    entity="cost-bo",
    project="memoised-realworld-exp",
    group=f"{args.exp_name}|-acqf_{args.acqf}|-dec-fac_{args.decay_factor}"
    f"|init-eta_{args.init_eta}",
    name=f"{date_now}-trial-number_{args.trial}",
    config=params,
)

consumed_budget, total_budget, init_budget = (
    0,
    params["total_budget"],
    params["budget_0"],
)

i = 0
warmup = True
if t5_init_dataset:
    dataset = t5_init_dataset["dataset"]
    consumed_budget = t5_init_dataset["consumed_budget"]
    n_init_data = i = t5_init_dataset["n_init_data"]
    warmup = False
try:
    while consumed_budget < total_budget:
        tic = time.time()

        if consumed_budget > init_budget and warmup:
            with s3.open(init_dataset_path, "wb") as f:
                t5_init_dataset = {
                    "dataset": dataset,
                    "consumed_budget": consumed_budget,
                    "n_init_data": i,
                }
                print("T5 HP Dataset")
                print(t5_init_dataset)
                pickle.dump(t5_init_dataset, f)
            warmup = False
            params["n_init_data"] = i
        print(hp_sampling_range)
        new_hp_dict, logging_metadata = generate_hps(
            dataset,
            hp_sampling_range,
            iteration=i,
            params=params,
            consumed_budget=consumed_budget,
            acq_type=args.acqf,
        )

        output_dir: str = args.cache_root / f"iter_{i}"
        s3.mkdirs(output_dir)
        with s3.open(output_dir, "wb") as output_dir_file:
            pipeline_outputs = t5_fine_tuning(data_dir, output_dir_file, new_hp_dict)
        obj, cost_per_stage = pipeline_outputs["obj"], pipeline_outputs["costs"]

        consumed_budget += sum(cost_per_stage)

        dataset = update_dataset_new_run(
            dataset,
            new_hp_dict,
            cost_per_stage,
            obj,
            logging_metadata["x_bounds"],
            args.acqf,
            device=device,
        )
        print("Stage costs:", cost_per_stage)
        print(
            f"\n\n[{time.strftime('%Y-%m-%d-%H%M')}]    Iteration-{i} [acq_type: {args.acqf}] Trial No. #{args.trial} Runtime: {time.time()-tic} Consumed Budget: {consumed_budget}"
        )
        eta = (total_budget - consumed_budget) / (total_budget - params["budget_0"])
        log_metrics(
            dataset,
            logging_metadata,
            args.exp_name,
            verbose=params["verbose"],
            iteration=i,
            trial=args.trial,
            acqf=args.acqf,
            eta=eta,
        )
        i += 1
finally:
    # Clean up cache
    if s3.exists(args.cache_root):
        s3.rmdir(args.cache_root, ignore_errors=True)
    wandb.finish()
