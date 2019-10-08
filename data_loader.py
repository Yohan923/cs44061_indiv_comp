import pandas as pd
import sklearn.preprocessing as skp

from constants import CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS


def load_data():
    data_set = dict()

    train_data = pd.read_csv("data/tcd ml 2019-20 income prediction training (with labels).csv", index_col="Instance")
    pred_data = pd.read_csv("data/tcd ml 2019-20 income prediction test (without labels).csv", index_col="Instance")
    train_data = train_data.fillna(train_data.mean()[NUMERIC_FEATURE_KEYS])
    pred_data = pred_data.fillna(pred_data.mean()[NUMERIC_FEATURE_KEYS])

    tmp = pd.concat([train_data, pred_data])

    for key in CATEGORICAL_FEATURE_KEYS:
        tmp[key] = pd.Categorical(tmp[key])
        tmp[key] = tmp[key].cat.codes
    tmp["Size of City"] = skp.scale(tmp["Size of City"])

    train_data = tmp.iloc[0:111993]
    pred_data = tmp.iloc[111993:]

    # generate dev and test sets
    dev_set = train_data.sample(11199)
    dev_index = dev_set.index
    for index in dev_index:
        train_data.drop(index, inplace=True)

    test_set = train_data.sample(11199)
    test_index = test_set.index
    for index in test_index:
        train_data.drop(index, inplace=True)

    data_set["train_Y"] = train_data.pop("Income")
    data_set["train_X"] = train_data

    data_set["dev_Y"] = dev_set.pop("Income")
    data_set["dev_X"] = dev_set

    data_set["test_Y"] = test_set.pop("Income")
    data_set["test_X"] = test_set

    data_set["pred_Y"] = pred_data.pop("Income")
    data_set["pred_X"] = pred_data

    return data_set
