import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder, RobustScaler

import data_loader
from constants import NUMERIC_FEATURE_KEYS, ONEHOT_CATEGORICAL_FEATURE_KEYS, LABEL_CATEGORICAL_FEATURE_KEYS


def preprocessing(data_X, data_Y=None, categories=[]):
    numerical_X = onehot_X = ordinal_X = None

    buffer = list()

    for (i, _) in NUMERIC_FEATURE_KEYS:
        buffer.append(data_X[:, i].reshape(-1, 1))

    numerical_X = np.concatenate(buffer, axis=1)

    buffer = list()
    for (i, _) in ONEHOT_CATEGORICAL_FEATURE_KEYS:
        buffer.append(data_X[:, i].reshape(-1, 1))

    onehot_X = np.concatenate(buffer, axis=1)

    buffer = list()
    for (i, _) in LABEL_CATEGORICAL_FEATURE_KEYS:
        buffer.append(data_X[:, i].reshape(-1, 1))

    ordinal_X = np.concatenate(buffer, axis=1)

    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    num_impute = SimpleImputer()
    numerical_X = num_impute.fit_transform(numerical_X)

    robust = RobustScaler()
    numerical_X = robust.fit_transform(numerical_X)

    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    cat_impute = SimpleImputer(strategy='constant')
    onehot_X = cat_impute.fit_transform(onehot_X)
    ordinal_X = cat_impute.fit_transform(ordinal_X)

    ordinal_enc = OrdinalEncoder()
    binary = ce.binary.BinaryEncoder(cols=[i for i in range(len(LABEL_CATEGORICAL_FEATURE_KEYS))])
    onehot = OneHotEncoder(categories=categories, handle_unknown='ignore')

    ordinal_X = ordinal_enc.fit_transform(ordinal_X)
    onehot_X = onehot.fit_transform(onehot_X).toarray()
    binary_ordinal_X = binary.fit_transform(ordinal_X)

    return np.concatenate([numerical_X, onehot_X, binary_ordinal_X], axis=1)


if __name__ == '__main__':
    data_set = data_loader.load_data()

    # data_set = data_loader.load_from_csv()

    X_train, X_test, y_train, y_test = train_test_split(
        data_set["train_X"].values, data_set["train_Y"].values, test_size=0.1, random_state=0)

    onehot_cats = list()
    for (i, _) in ONEHOT_CATEGORICAL_FEATURE_KEYS:
        cat_impute = SimpleImputer(strategy='constant')
        X_train[:, i] = cat_impute.fit_transform(X_train[:, i].reshape(-1, 1)).reshape(-1)
        onehot = OneHotEncoder()
        onehot_model = onehot.fit(X_train[:, i].reshape(-1, 1))
        onehot_cats.append(onehot_model.categories_)

    tmp_l = list()
    for l in onehot_cats:
        tmp_l.append(l[0].tolist())

    X_train = preprocessing(X_train, categories=tmp_l)
    X_test = preprocessing(X_test, categories=tmp_l)
    X_pred = preprocessing(data_set["pred_X"].values, categories=tmp_l)

    # poly = ('poly', PolynomialFeatures(degree=2))
    linear = LinearRegression(fit_intercept=False, normalize=False)

    model = linear

    model = model.fit(X_train, y_train)

    test = model.predict(X_test)

    error = (1 / 2 * (len(y_test))) * np.sum(np.square(test - y_test))

    pred = model.predict(X_pred)

    pd.DataFrame(pred).to_csv("testing2.csv")

    print("finished")
