import pandas as pd


def select_features(data: pd.DataFrame) -> pd.Series:
    categorical_feats = [f for f in data.columns if data[f].dtype == "object"]

    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])

    return data


def get_avg_prev(prev: pd.DataFrame) -> pd.DataFrame:
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == "object"]
    for f_ in prev_cat_features:
        prev[f_], _ = pd.factorize(prev[f_])

    avg_prev = prev.groupby("SK_ID_CURR").mean()
    cnt_prev = prev[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
    avg_prev["nb_app"] = cnt_prev["SK_ID_PREV"]

    avg_prev.drop("SK_ID_PREV", inplace=True, axis=1)

    return avg_prev


def prepare_data(data, prev) -> pd.DataFrame:
    data = select_features(data)

    avg_prev = get_avg_prev(prev)
    data = data.merge(right=avg_prev.reset_index(), how="left", on="SK_ID_CURR")

    data = data.fillna(0)

    excluded_feats = ["SK_ID_CURR"]
    features = [f_ for f_ in data.columns if f_ not in excluded_feats]
    data = data[features]

    return data
