import pandas as pd
import sklearn.preprocessing as skp

from constants import CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS


def load_data():
    data_set = dict()

    train_data = pd.read_csv("data/tcd ml 2019-20 income prediction training (with labels).csv", index_col="Instance",
                             na_values={'Gender': ["no", '0', "other", 'unknown'], 'University Degree': ["No", "0"],
                                        'Hair Color': ["unknown", "0", "Unknown"]})
    pred_data = pd.read_csv("data/tcd ml 2019-20 income prediction test (without labels).csv", index_col="Instance",
                            na_values={'Gender': ["no", '0', 'other', 'Unknown'], 'University Degree': ["No", "0"],
                                       'Hair Color': ["unknown", "0", "Unknown"]})
    """
    train_data = train_data.fillna(train_data.mean()[NUMERIC_FEATURE_KEYS])
    pred_data = pred_data.fillna(pred_data.mean()[NUMERIC_FEATURE_KEYS])

    tmp = pd.concat([train_data, pred_data])

    for key in CATEGORICAL_FEATURE_KEYS:
        tmp[key] = pd.Categorical(tmp[key])
        tmp[key] = tmp[key].cat.codes
    tmp["Size of City"] = skp.scale(tmp["Size of City"])

    train_data = tmp.iloc[0:111993]
    pred_data = tmp.iloc[111993:]
    """

    data_set["train_Y"] = train_data.pop("Income")
    data_set["train_X"] = train_data

    data_set["pred_Y"] = pred_data.pop("Income")
    data_set["pred_X"] = pred_data

    data_set["train_Y"].to_csv("data/trainY.csv")
    data_set["train_X"].to_csv("data/trainX.csv")

    data_set["pred_Y"].to_csv("data/predY.csv")
    data_set["pred_X"].to_csv("data/predX.csv")

    return data_set


def load_from_csv():
    data_set = dict()

    data_set["train_Y"] = pd.read_csv("data/trainY.csv", names=["Income"])
    data_set["train_X"] = pd.read_csv("data/trainX.csv", index_col="Instance")

    data_set["pred_Y"] = pd.read_csv("data/predY.csv", names=["Income"])
    data_set["pred_X"] = pd.read_csv("data/predX.csv", index_col="Instance")

    return data_set
