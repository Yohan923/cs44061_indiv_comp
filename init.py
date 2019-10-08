from sklearn import linear_model
import numpy as np
import joblib as jb
import pandas as pd
import sklearn.preprocessing as skp

from constants import FEATURE_COLUMNS, CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS

if __name__ == '__main__':
    data = pd.read_csv("data/tcd ml 2019-20 income prediction training (with labels).csv", index_col="Instance")
    data.fillna(data.mean()[NUMERIC_FEATURE_KEYS], inplace=True)
    Y = data.pop('Income')

    for key in CATEGORICAL_FEATURE_KEYS:
        data[key] = pd.Categorical(data[key])
        data[key] = data[key].cat.codes

    data["Size of City"] = skp.scale(data["Size of City"])

    tmp = data.values

    reg = linear_model.LinearRegression()

    reg.fit(tmp, Y.values)

    data2 = pd.read_csv("data/tcd ml 2019-20 income prediction test (without labels).csv", index_col="Instance")
    data2.fillna(data2.mean()[NUMERIC_FEATURE_KEYS], inplace=True)
    Y = data2.pop('Income')

    for key in CATEGORICAL_FEATURE_KEYS:
        data2[key] = pd.Categorical(data2[key])
        data2[key] = data2[key].cat.codes

    data2["Size of City"] = skp.scale(data2["Size of City"])

    tmp2 = reg.predict(data2.values)

    p = pd.DataFrame(tmp2)
    p.to_csv("testing.csv")

    print("finished")
