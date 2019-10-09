from sklearn import linear_model
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import data_loader

from constants import FEATURE_COLUMNS, CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS

if __name__ == '__main__':

    data_set = data_loader.load_data()
    #data_set = data_loader.load_from_csv()

    reg = linear_model.LinearRegression()

    reg.fit(data_set["train_X"].values, data_set["train_Y"].values)

    tmp = reg.predict(data_set["pred_X"].values)

    p = pd.DataFrame(tmp)
    p.to_csv("testing1.csv")

    print("finished")
