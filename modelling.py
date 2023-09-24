import json
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from cachestore import Cache, Formatter
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

NFOLDS = 3
SEED = 0


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
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

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
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


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
    y_test = test["TARGET"]
    x_train = train.drop("TARGET", axis=1)
    x_test = test.drop("TARGET", axis=1)
    
    # Train the meta modles.
    xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test)
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
    
    x_train = np.concatenate((xg_oof_train, et_oof_train), axis=1)
    x_test = np.concatenate((xg_oof_test, et_oof_test), axis=1)
    
    return {"train": pd.DataFrame(x_train, columns=["xg", "et"]), "test": pd.DataFrame(x_test, columns=["xg", "et"])}
    
def train_basemodels2(
    train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    # Initiate the meta models.
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=params["rf_params"])
    cb = CatboostWrapper(clf=CatBoostClassifier, seed=SEED, params=params["catboost_params"])

    # Separate features from targets.
    y_train = train["TARGET"]
    y_test = test["TARGET"]
    x_train = train.drop("TARGET", axis=1)
    x_test = test.drop("TARGET", axis=1)

    # Train the meta modles.
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
    cb_oof_train, cb_oof_test = get_oof(cb, x_train, y_train, x_test)

    x_train = np.concatenate((rf_oof_train, cb_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, cb_oof_test), axis=1)

    return {"train": pd.DataFrame(x_train, columns=["rf", "cb"]), "test": pd.DataFrame(x_test, columns=["rf", "cb"])}


def train_metamodel(
    train: np.ndarray,
    # train_targets: np.ndarray,
    test: np.ndarray,
    # test_targets: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    logistic_regression = LogisticRegression(**params["lr_params"], random_state=SEED)
    y_train = train.pop("TARGET")
    y_test = test.pop("TARGET")
    logistic_regression.fit(train, y_train)

    test_preds = logistic_regression.predict_proba(test)[:, 1]
    au_roc = roc_auc_score(y_test, test_preds)

    return {"AUROC": au_roc}



sys.path.append("./")
from data_processing import prepare_data

parser = ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True, help="Specifies a unique experiment name")
parser.add_argument("--trial", type=int, help="The trial number", default=1)
parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
parser.add_argument("--decay-factor", type=float, help="Decay factor", default=0.95)
parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPS", "CArBO", "EI", "RAND"], default="EI")
parser.add_argument("--cache-root", type=Path, default=".cachestore", help="Cache directory")
parser.add_argument("--disable-cache", action="store_true", help="Disable cache")
parser.add_argument("--data-dir", type=Path, help="Directory with the data", default="/home/ridwan/workdir/cost_aware_bo/inputs/")
args = parser.parse_args()

cache = Cache(disable=args.disable_cache)

data_dir = args.data_dir

def preprocess():        
    train = pd.read_csv(data_dir/"train.csv")
    test = pd.read_csv(data_dir/"test.csv")
    prev = pd.read_csv(data_dir/"previous_application.csv")
    print(f"Train shape: {train.shape} ::: Test shape: {test.shape} ::: Prev shape: {prev.shape}")
    print(f"Class dist train: {train['TARGET'].value_counts()}; test: {test['TARGET'].value_counts()}")
    
    train = prepare_data(train, prev).sample(frac=0.01, ignore_index=True, random_state=SEED)
    test = prepare_data(test, prev).sample(frac=0.01, ignore_index=True, random_state=SEED)

    print(f"Shape after prep: train-{train.shape} :: test-{test.shape}")

    return train, test

train, test = preprocess()

# Train base models.
@cache(ignore={"train", "test"})
def stage_1(train, test, hp1_params):
    base_model_op1 = train_basemodels1(train, test, hp1_params)
    base_model_op2 = train_basemodels2(train, test, hp1_params)
    train = pd.concat([base_model_op1["train"], base_model_op2["train"], train["TARGET"]], axis=1)
    test = pd.concat([base_model_op1["test"], base_model_op2["test"], test["TARGET"]], axis=1)
    return train, test

# Train the meta model and generate test AUROC score.
# @cache(ignore={"train", "test"})
def stage_2(train, test, hp2_params):
    meta_model_op = train_metamodel(train, test, hp2_params)
    with args.obj_output.open("w") as f:
        json.dump(meta_model_op, f)
    return meta_model_op

def main(new_hps_dict):
    hp1_params={
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
        }
    }
    
    hp2_params = {
        "C": new_hps_dict["1__C"],
        "tol": new_hps_dict["1__tol"],
        "max_iter": new_hps_dict["1__max_iter"]
    }
    base0_time = time.time()
    train, test = stage_1(train, test, hp1_params)
    base1_time = time.time()

    print(f"Base1 duration: {base1_time-base0_time}")

    auroc_dict = stage_2(train, test, hp2_params)
    obj = auroc_dict["AUROC"]
    meta_time = time.time()

    cost_per_stage = [base1_time, meta_time]
    print(f"Meta duration: {meta_time-base1_time}")
    
    return obj, cost_per_stage

dataset = {}
with open("segmentation/sampling-range.json") as f:
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
    "total_budget": 15000,
    "budget_0": 7000
}

args_dict = deepcopy(vars(args))
params.update(args_dict)