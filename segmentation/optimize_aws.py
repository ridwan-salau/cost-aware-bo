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

from segmentation_exp import (  # Importing DroneDataset to prevent pickle error
    main,
    DroneDataset # noqa
)
import s3fs

s3 = s3fs.S3FileSystem()

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
    choices=["EEIPU", "MS_CArBO", "EIPS", "CArBO", "EI", "RAND", "LaMBO", "MS_BO"],
    default="EI",
)
parser.add_argument(
    "--cache-root", type=Path, default=".cachestore", help="Cache directory"
)
parser.add_argument("--disable-cache", action="store_true", help="Disable cache")
parser.add_argument(
    "--data-dir", type=Path, help="Directory with the data", default="./inputs"
)
args, _ = parser.parse_known_args()

data_dir: Path = args.data_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

root = "mbz-hpc-aws-master/AROARU6TOWKRU3FNVE2PB:Ridwan.Salahuddeen@mbzuai.ac.ae/segmentation"
init_dataset_path = Path(
    f"{root}/inputs/{args.exp_name}/stacking_init_dataset-trial_{args.trial}.pk"
)
s3.mkdirs(init_dataset_path.parent, exist_ok=True)
dataset = {}
stacking_init_dataset = {}
if s3.exists(init_dataset_path):
    with s3.open(init_dataset_path, "rb") as f:
        segment_init_dataset = pickle.load(f)

with (data_dir / "initial_hparams_multi.json").open() as f:
    initial_hparams = json.load(f)
    hp_sampling_range = initial_hparams["hp_sampling_range"]
    params = initial_hparams["params"]

args_dict = deepcopy(vars(args))
params.update(args_dict)

date_now = f"{time.strftime('%Y-%m-%d-%H%M')}"

wandb.init(
    entity="cost-bo",
    project="jan-2024-cost-aware-bo",
    group=f"{args.exp_name}|-acqf_{args.acqf}|-dec-fac_{args.decay_factor}"
    f"|init-eta_{args.init_eta}",
    name=f"{date_now}-trial-number_{args.trial}",
    config=params,
)

consumed_budget, total_budget, n_init_data = (
    0,
    params["total_budget"],
    params["n_init_data"],
)

i = 0
warmup = True
if segment_init_dataset:
    dataset = segment_init_dataset["dataset"]
    params["budget_0"] = consumed_budget = segment_init_dataset["consumed_budget"]
    i = segment_init_dataset["n_init_data"]
    warmup = False
try:
    while consumed_budget < total_budget:
        tic = time.time()

        if i >= n_init_data and warmup:  # Only execute this for the once for a trial
            with init_dataset_path.open("wb") as f:
                segment_init_dataset = {
                    "dataset": dataset,
                    "consumed_budget": consumed_budget,
                    "n_init_data": i,
                }
                print("Segmentation HP Dataset")
                print(segment_init_dataset)
                pickle.dump(segment_init_dataset, f)
            warmup = False
            params["budget_0"] = consumed_budget
        print(hp_sampling_range)
        new_hp_dict, logging_metadata = generate_hps(
            dataset,
            hp_sampling_range,
            iteration=i,
            params=params,
            consumed_budget=consumed_budget,
            acq_type=args.acqf,
            trial=args.trial,
            exp_name=args.exp_name,
        )

        output_dir: Path = args.cache_root / f"iter_{i}"
        output_dir.mkdir(parents=True)
        pipeline_outputs = main(new_hp_dict)
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
        eta = (
            1
            if i < n_init_data
            else (total_budget - consumed_budget) / (total_budget - params["budget_0"])
        )
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
    if os.path.exists(args.cache_root):
        shutil.rmtree(args.cache_root, ignore_errors=True)
    wandb.finish()
