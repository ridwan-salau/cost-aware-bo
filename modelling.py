from math import sqrt
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

NFOLDS = 3
SEED = 0
NROWS = None


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

    return {"AU_ROC": au_roc}

if __name__=="__main__":
    from pathlib import Path
    import json
    import sys
    import time
    from cachestore import Cache, Formatter  
    from argparse import ArgumentParser
    
    sys.path.append("./")
    from data_processing import prepare_data
    
    parser = ArgumentParser()
    parser.add_argument("--stage-costs-outputs", nargs='+', type=Path, help="Stage costs outputs directory")
    parser.add_argument("--obj-output", type=Path, help="The output directory of the objective.", required=True)
    parser.add_argument("--disable-cache", action="store_true", help="Disable cache")
    
    args = parser.parse_args()
    
    cache = Cache(disable=args.disable_cache)
    
    data_dir = Path("/home/ridwan/workdir/cost_aware_bo/inputs/")
    
    @cache()
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
    
    st = time.time()
    train, test = preprocess()
    prep_time = time.time()
    
    print(f"Prep duration: {prep_time-st}")

    hp1 = json.load((data_dir/"stage1.json").open("r"))[0]
    # Train base models.
    @cache(ignore={"train", "test"})
    def stage_1(train, test, hp1):
        base_model_op1 = train_basemodels1(train, test, hp1)
        return base_model_op1
    
    base_model_op1 = stage_1(train, test, hp1)
    base1_time = time.time()
    
    print(f"Base1 duration: {base1_time-prep_time}")
    
    hp2 = json.load((data_dir/"stage2.json").open("r"))[1]
    
    @cache(ignore={"train", "test"})
    def stage_2(train, test, hp2):
        base_model_op2 = train_basemodels2(train, test, hp2)
        train = pd.concat([base_model_op1["train"], base_model_op2["train"], train["TARGET"]], axis=1)
        test = pd.concat([base_model_op1["test"], base_model_op2["test"], test["TARGET"]], axis=1)
        return train, test
    
    train, test = stage_2(train, test, hp2)
    base2_time = time.time()
    
    print(f"Base2 duration: {base2_time-base1_time}")
    
    hp3 = json.load((data_dir/"stage3.json").open("r"))[2]
    # Train the meta model and generate test AU-ROC score.
    @cache(ignore={"train", "test"})
    def stage_3(train, test, hp3):
        meta_model_op = train_metamodel(train, test, hp3)
        with args.obj_output.open("w") as f:
            json.dump(meta_model_op, f)
        return meta_model_op
    
    meta_model_op = stage_3(train, test, hp3)
    meta_time = time.time()
    
    print(f"Meta duration: {meta_time-base2_time}")
    
    print(meta_model_op)
    
    print(f"Base1 duration: {base1_time-prep_time}")
    print(f"Base2 duration: {base2_time-base1_time}")
    print(f"Meta duration: {meta_time-base2_time}")
    print(f"Total duration: {meta_time-st}")
    
    per_stg_dur = [base1_time-prep_time, base2_time-base1_time, meta_time-base2_time]
    for stage, wall_time in zip(args.stage_costs_outputs, per_stg_dur):
        print(f"Saving {stage}, {wall_time}")
        with stage.open("w") as f:
            json.dump({"wall_time":wall_time}, f)
        
