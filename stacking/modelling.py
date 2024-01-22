import json
import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from cost_aware_bo import generate_hps, log_metrics, update_dataset_new_run
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import wandb
from cachestore import Cache, LocalStorage

from data_processing import prepare_data

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
args = parser.parse_args()
disable_cache = args.acqf != "EEIPU"
cache = Cache(
    f"stacking_{args.exp_name}_{args.trial}_cache",
    storage=LocalStorage(args.cache_root),
)

data_dir: Path = args.data_dir

NFOLDS = 3
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_state"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_seed"] = seed
        self.clf: CatBoostClassifier = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train, silent=True)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param["random_state"] = seed
        self.nrounds = params.pop("nrounds", 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x, silent=True))


def get_oof(clf, x_train, y_train, x_test, n_folds=3):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((n_folds, x_test.shape[0]))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.loc[train_index]
        y_tr = y_train.loc[train_index]
        x_te = x_train.loc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def train_basemodels1(
    train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    # Initiate the meta models.
    xg = XgbWrapper(seed=SEED, params=params["xgb_params"])
    et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=params["et_params"])

    # Separate features from targets.
    y_train = train["TARGET"]
    # y_test = test["TARGET"]
    x_train = train.drop("TARGET", axis=1)
    x_test = test.drop("TARGET", axis=1)

    # Train the meta modles.
    xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test)
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

    x_train = np.concatenate((xg_oof_train, et_oof_train), axis=1)
    x_test = np.concatenate((xg_oof_test, et_oof_test), axis=1)

    return {
        "train": pd.DataFrame(x_train, columns=["xg", "et"]),
        "test": pd.DataFrame(x_test, columns=["xg", "et"]),
    }


def train_basemodels2(
    train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    # Initiate the meta models.
    rf = SklearnWrapper(
        clf=RandomForestClassifier, seed=SEED, params=params["rf_params"]
    )
    cb = CatboostWrapper(
        clf=CatBoostClassifier, seed=SEED, params=params["catboost_params"]
    )

    # Separate features from targets.
    y_train = train["TARGET"]
    # y_test = test["TARGET"]
    x_train = train.drop("TARGET", axis=1)
    x_test = test.drop("TARGET", axis=1)

    # Train the meta modles.
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
    cb_oof_train, cb_oof_test = get_oof(cb, x_train, y_train, x_test)

    x_train = np.concatenate((rf_oof_train, cb_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, cb_oof_test), axis=1)

    return {
        "train": pd.DataFrame(x_train, columns=["rf", "cb"]),
        "test": pd.DataFrame(x_test, columns=["rf", "cb"]),
    }


def train_metamodel(
    train: np.ndarray,
    # train_targets: np.ndarray,
    test: np.ndarray,
    # test_targets: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    logistic_regression = LogisticRegression(**params, random_state=SEED)
    y_train = train.pop("TARGET")
    y_test = test.pop("TARGET")
    logistic_regression.fit(train, y_train)

    test_preds = logistic_regression.predict_proba(test)[:, 1]
    au_roc = roc_auc_score(y_test, test_preds)

    return {"AUROC": au_roc}


@cache()
def preprocess():
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    prev = pd.read_csv(data_dir / "previous_application.csv")
    print(
        f"Train shape: {train.shape} ::: Test shape: {test.shape} ::: Prev shape: {prev.shape}"
    )
    print(
        f"Class dist train: {train['TARGET'].value_counts()}; test: {test['TARGET'].value_counts()}"
    )

    train = prepare_data(train, prev).sample(
        frac=0.01, ignore_index=True, random_state=SEED
    )
    test = prepare_data(test, prev).sample(
        frac=0.01, ignore_index=True, random_state=SEED
    )

    print(f"Shape after prep: train-{train.shape} :: test-{test.shape}")

    return train, test


train, test = preprocess()


# Train base models.
@cache(ignore={"train", "test"}, disable=disable_cache)
def stage_1(train, test, hp1_params):
    base_model_op1 = train_basemodels1(train, test, hp1_params)
    print("Done with basemodel 1")
    base_model_op2 = train_basemodels2(train, test, hp1_params)
    train = pd.concat(
        [base_model_op1["train"], base_model_op2["train"], train["TARGET"]], axis=1
    )
    test = pd.concat(
        [base_model_op1["test"], base_model_op2["test"], test["TARGET"]], axis=1
    )
    return train, test


# Train the meta model and generate test AUROC score.
# @cache(ignore={"train", "test"})
def stage_2(train, test, hp2_params):
    meta_model_op = train_metamodel(train, test, hp2_params)
    # with args.obj_output.open("w") as f:
    #     json.dump(meta_model_op, f)
    return meta_model_op


def main(new_hps_dict):
    hp1_params = {
        "et_params": {
            "n_estimators": new_hps_dict["0__n_estimators0"],
        },
        "xgb_params": {
            "learning_rate": new_hps_dict["0__learning_rate0"],
            "max_depth": new_hps_dict["0__max_depth0"],
        },
        "rf_params": {
            "n_estimators": new_hps_dict["0__n_estimators1"],
            "max_depth": new_hps_dict["0__max_depth1"],
        },
        "catboost_params": {
            "learning_rate": new_hps_dict["0__learning_rate1"],
        },
    }

    hp2_params = {
        "C": new_hps_dict["1__C"],
        "tol": new_hps_dict["1__tol"],
        "max_iter": new_hps_dict["1__max_iter"],
    }
    base0_time = time.time()
    train_stg1, test_stg1 = stage_1(train, test, hp1_params)
    base1_time = time.time()

    print(f"Base1 duration: {base1_time-base0_time}")

    auroc_dict = stage_2(train_stg1, test_stg1, hp2_params)
    obj = auroc_dict["AUROC"]
    meta_time = time.time()

    cost_per_stage = [base1_time - base0_time, meta_time - base1_time]
    print(f"Meta duration: {meta_time-base1_time}")

    return obj, cost_per_stage


dataset = {}
with open("inputs/sampling-range-stacking.json") as f:
    hp_sampling_range = json.load(f)

params = {
    # "decay_factor": 0.95,
    # "init_eta": 0.05,
    "n_trials": 10,
    "n_iters": 40,
    "alpha": 9,
    "norm_eps": 1,
    "epsilon": 0.1,
    "batch_size": 1,
    "normalization_bounds": [0, 1],
    "cost_samples": 1000,
    "n_init_data": 10,
    "prefix_thresh": 10000000,
    "warmup_iters": 10,
    "use_pref_pool": 1,
    "verbose": 1,
    "rand_seed": 42,
    "total_budget": 500,
    # "budget_0": 400
    "n_prefixes": 5,
}

init_dataset_path = Path(
    f"inputs/{args.exp_name}/stacking_init_dataset-trial_{args.trial}.pk"
)
init_dataset_path.parent.mkdir(parents=True, exist_ok=True)
dataset = {}
stacking_init_dataset = {}
if init_dataset_path.exists():
    with init_dataset_path.open("rb") as f:
        stacking_init_dataset = pickle.load(f)

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
if stacking_init_dataset:
    dataset = stacking_init_dataset["dataset"]
    params["budget_0"] = consumed_budget = stacking_init_dataset["consumed_budget"]
    i = stacking_init_dataset["n_init_data"]
    warmup = False
try:
    while consumed_budget < total_budget:
        tic = time.time()

        if i >= n_init_data and warmup:  # Only execute this for the once for a trial
            with init_dataset_path.open("wb") as f:
                stacking_init_dataset = {
                    "dataset": dataset,
                    "consumed_budget": consumed_budget,
                    "n_init_data": i,
                }
                print("Stacking HP Dataset")
                print(stacking_init_dataset)
                pickle.dump(stacking_init_dataset, f)
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
        )

        output_dir: Path = args.cache_root / f"iter_{i}"
        output_dir.mkdir(parents=True)

        obj, cost_per_stage = main(new_hp_dict)

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
