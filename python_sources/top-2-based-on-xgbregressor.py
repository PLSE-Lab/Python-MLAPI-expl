#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


# In[ ]:


X_train = pd.read_csv("../input/rossmann-store-sales/train.csv", parse_dates=[2], low_memory=False)
X_test = pd.read_csv("../input/rossmann-store-sales/test.csv", parse_dates=[3], low_memory=False)
store = pd.read_csv("../input/rossmann-store-sales/store.csv")


# In[ ]:


print(X_train["Date"].min(), X_train["Date"].max())
X_train.sort_values(["Date"], inplace=True, kind="mergesort")
X_train.reset_index(drop=True, inplace=True)


# In[ ]:


X_test["Open"] = X_test["Open"].fillna(1)


# In[ ]:


# TODO: how do handle missing values regarding competitor store
store.fillna(0, inplace=True)


# In[ ]:


X_train = pd.merge(X_train, store, on="Store", how="left")
X_test = pd.merge(X_test, store, on="Store", how="left")


# In[ ]:


for df in [X_train, X_test]:
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    assert np.all(df["DayOfWeek"] - 1 == df['Date'].dt.dayofweek)
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.weekofyear
    df.drop("Date", axis=1, inplace=True)


# In[ ]:


for df in [X_train, X_test]:
    df["CompetitionOpen"] = ((df["year"] - df["CompetitionOpenSinceYear"]) * 12
                             + (df["month"] - df["CompetitionOpenSinceMonth"]))
    df["CompetitionOpen"] = df["CompetitionOpen"].apply(lambda x: x if x > 0 else 0)
    df["PromoOpen"] = ((df["year"] - df["Promo2SinceYear"]) * 12
                       + (df["weekofyear"] - df["Promo2SinceWeek"]) / 4)
    df["PromoOpen"] = df["PromoOpen"].apply(lambda x: x if x > 0 else 0)


# In[ ]:


month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
def check(row):
    if isinstance(row['PromoInterval'], str) and month2str[row['month']] in row['PromoInterval']:
        if (row['year'] > row['Promo2SinceYear'] or
            (row['year'] == row['Promo2SinceYear'] and row['weekofyear'] > row['Promo2SinceWeek'])):
            return 1
    return 0
for df in [X_train, X_test]:
    df['IsPromoMonth'] = df.apply(lambda row: check(row), axis=1) 
    #df.drop("PromoInterval", axis=1, inplace=True)


# In[ ]:


groups = X_train[["Store", "Open"]].groupby("Store").mean()
groups.rename(columns={"Open":"shopavgopen"}, inplace=True)
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


groups = X_train[["Store", "Sales", "Customers"]].groupby("Store").sum()
groups["shopavgsalespercustomer"] = groups["Sales"] / groups["Customers"]
del groups["Sales"], groups["Customers"]
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


groups = X_train[["Store", "SchoolHoliday"]].groupby("Store").mean()
groups.rename(columns={"SchoolHoliday":"shopavgschoolholiday"}, inplace=True)
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


groups1 = X_train[["Store", "Sales"]].groupby("Store").sum()
groups2 = X_train[X_train["StateHoliday"] != "0"][["Store", "Sales"]].groupby("Store").sum()
groups = pd.merge(groups1, groups2, on="Store")
groups["shopsalesholiday"] = groups["Sales_y"] / groups["Sales_x"]
del groups["Sales_x"], groups["Sales_y"]
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


groups1 = X_train[["Store", "Sales"]].groupby("Store").sum()
groups2 = X_train[X_train["IsPromoMonth"] == 1][["Store", "Sales"]].groupby("Store").sum()
groups = pd.merge(groups1, groups2, on="Store")
groups["shopsalespromo"] = groups["Sales_y"] / groups["Sales_x"]
del groups["Sales_x"], groups["Sales_y"]
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


groups1 = X_train[["Store", "Sales"]].groupby("Store").sum()
groups2 = X_train[X_train["DayOfWeek"] == 6][["Store", "Sales"]].groupby("Store").sum()
groups = pd.merge(groups1, groups2, on="Store")
groups["shopsalessaturday"] = groups["Sales_y"] / groups["Sales_x"]
del groups["Sales_x"], groups["Sales_y"]
X_train = pd.merge(X_train, groups, how="left", on="Store")
X_test = pd.merge(X_test, groups, how="left", on="Store")


# In[ ]:


assert np.all(X_train[X_train["Open"] == 0]["Sales"] == 0)
# X_train = X_train[X_train["Open"] == 1]
X_train = X_train[X_train["Sales"] != 0]
del X_train["Open"]
test_close_ind = np.where(X_test["Open"] == 0)[0]
del X_test["Open"]


# In[ ]:


for col in ["StateHoliday", "StoreType", "Assortment", "DayOfWeek", "month", "PromoInterval"]:
    for val in X_train[col].unique():
        new_col_name = col + "_" + str(val)
        X_train[new_col_name] = (X_train[col] == val).astype(int)
        X_test[new_col_name] = (X_test[col] == val).astype(int)
del X_train["PromoInterval"], X_test["PromoInterval"]


# In[ ]:


for col in ["StateHoliday", "StoreType", "Assortment"]:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])


# In[ ]:


y_train = np.array(X_train["Sales"])
X_train.drop("Sales", axis=1, inplace=True)
X_train.drop("Customers", axis=1, inplace=True)
test_ID = X_test["Id"]
X_test.drop("Id", axis=1, inplace=True)


# In[ ]:


y_train = np.log1p(y_train)


# In[ ]:


def rmspe(y_true, y_pred):
    y_pred = y_pred[y_true != 0]
    y_true = y_true[y_true != 0]
    err = np.sqrt(np.mean((1 - y_pred / y_true) ** 2))
    return err

def rmspe_xgb(y_pred, y_true):
    y_true = y_true.get_label()
    err = rmspe(np.expm1(y_true), np.expm1(y_pred))
    return "rmspe", err


# In[ ]:


valid_mask = (X_train["year"] == 2015) & (X_train["dayofyear"] >= 171)  # last 6 weeks
X_train_1, y_train_1 = X_train[~valid_mask], y_train[~valid_mask]
X_train_2, y_train_2 = X_train[valid_mask], y_train[valid_mask]
reg = xgb.XGBRegressor(n_estimators=5000, objective="reg:squarederror", max_depth=10,
                       learning_rate=0.03, colsample_bytree=0.7, subsample=0.9,
                       random_state=0, tree_method="gpu_hist")
reg.fit(X_train_1, y_train_1, eval_set=[(X_train_1, y_train_1), (X_train_2, y_train_2)],
        eval_metric=rmspe_xgb, early_stopping_rounds=100, verbose=100)
best_iteration = reg.best_iteration


# In[ ]:


pred = np.expm1(reg.predict(X_test))
pred[test_close_ind] = 0
submission = pd.DataFrame({"Id": test_ID, "Sales": pred},
                          columns=["Id", "Sales"])
# 0.11690 0.11939
submission.to_csv("submission_1.csv", index=False)


# In[ ]:


pred = np.expm1(reg.predict(X_test) * 0.995)
pred[test_close_ind] = 0
submission = pd.DataFrame({"Id": test_ID, "Sales": pred},
                          columns=["Id", "Sales"])
# 0.11392 0.11196
submission.to_csv("submission_2.csv", index=False)


# In[ ]:


X_train, y_train = X_train[y_train != 0], y_train[y_train != 0]
reg = xgb.XGBRegressor(n_estimators=best_iteration, objective="reg:squarederror", max_depth=10,
                       learning_rate=0.03, colsample_bytree=0.7, subsample=0.9,
                       random_state=0, tree_method="gpu_hist")
reg.fit(X_train, y_train, eval_set=[(X_train, y_train)],
        eval_metric=rmspe_xgb, early_stopping_rounds=100, verbose=100)


# In[ ]:


pred = np.expm1(reg.predict(X_test))
pred[test_close_ind] = 0
submission = pd.DataFrame({"Id": test_ID, "Sales": pred},
                          columns=["Id", "Sales"])
# 0.11357 0.10751
submission.to_csv("submission_3.csv", index=False)


# In[ ]:


pred = np.expm1(reg.predict(X_test) * 0.995)
pred[test_close_ind] = 0
submission = pd.DataFrame({"Id": test_ID, "Sales": pred},
                          columns=["Id", "Sales"])
# 0.11245 0.10437
submission.to_csv("submission_4.csv", index=False)


# In[ ]:


pred = np.zeros(X_test.shape[0])
n_models = 5
for i in range(n_models):
    print("=== model " + str(i) + " ===")
    reg = xgb.XGBRegressor(n_estimators=5000, objective="reg:squarederror", max_depth=10,
                           learning_rate=0.03, colsample_bytree=0.7, subsample=0.9,
                           random_state=i, tree_method="gpu_hist")
    reg.fit(X_train_1, y_train_1, eval_set=[(X_train_1, y_train_1), (X_train_2, y_train_2)],
            eval_metric=rmspe_xgb, early_stopping_rounds=100, verbose=100)
    best_iteration = reg.best_iteration
    reg = xgb.XGBRegressor(n_estimators=best_iteration, objective="reg:squarederror", max_depth=10,
                           learning_rate=0.03, colsample_bytree=0.7, subsample=0.9,
                           random_state=i, tree_method="gpu_hist")
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train)],
            eval_metric=rmspe_xgb, early_stopping_rounds=100, verbose=100)
    pred += np.expm1(reg.predict(X_test) * 0.995)
pred /= n_models
pred[test_close_ind] = 0
submission = pd.DataFrame({"Id": test_ID, "Sales": pred},
                           columns=["Id", "Sales"])
# 0.11180 0.10257
submission.to_csv("submission_5.csv", index=False)

