# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/train.csv")
processed_data = pd.get_dummies(data)

import xgboost
from sklearn.model_selection import GridSearchCV

data_y = processed_data["SalePrice"]
data_X = processed_data.drop(["SalePrice","Id"],axis=1)

XGBR = xgboost.XGBRFRegressor()
XGBR_grid = {
    "min_chile_weight": [0.5,1,2,5],
    "max_depth": [2,3,5,10],
    "gamma": [0,0.1,0.2,0.5],
    "max_delta_step": [0,1],
    "reg_lambda": [1,2,5,10,20,50],
    "n_estimators": [50,100,150]
}
gsXGBR = GridSearchCV(XGBR,param_grid=XGBR_grid,n_jobs=-1,cv=5)
gsXGBR.fit(data_X,data_y)
XGBR_best = gsXGBR.best_score_
XGBR = gsXGBC.best_estimator_
submit_data = pd.read_csv("../input/test.csv")
submit_X = pd.get_dummies(submit_data)
submit_X = submit_X.drop(["Id"],axis=1)

def fill_col(column):
    col_name = column.name
    global cnt
    global submit_X
    if not (col_name in submit_X.columns):
        submit_X[col_name] = 0
        print(col_name)
    
data_X.apply(fill_col,axis=0)

submit_X = submit_X[data_X.columns]
predict_y = XGBR.predict(submit_X)
result = pd.DataFrame({"Id":submit_data["Id"],"SalePrice":predict_y})
result.to_csv("../input/result.csv",index=False)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.