# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def getMea(p_max_leaf_nodes,p_train_x,p_val_x,p_train_y,p_val_y):
    fun_model=DecisionTreeRegressor(max_leaf_nodes=p_max_leaf_nodes,random_state=0)
    fun_model.fit(p_train_x,p_train_y)
    fun_predict_val=fun_model.predict(p_val_x)
    fun_mea = mean_absolute_error(fun_predict_val,p_val_y)
    return fun_mea


train_data = pd.read_csv("../input/train.csv")

#print("train data columns ")

#print(train_data.columns)

y=train_data.SalePrice

train_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

x =train_data[train_predictors]

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)


train_model = DecisionTreeRegressor()

train_model.fit(train_x,train_y)

val_predict = train_model.predict(val_x)

mea_val=mean_absolute_error(val_predict,val_y)

for v_max_leaf_nodes in [5,20,30,35,40,45,50,60,100,200,500,5000]:
    mea_val = getMea(v_max_leaf_nodes,train_x,val_x,train_y,val_y)
    print("For Max Leaf Nodes %d the mean absoloute error is %d"%(v_max_leaf_nodes,mea_val))
