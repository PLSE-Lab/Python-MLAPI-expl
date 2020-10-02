# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))
dftrain=pd.read_csv("../input/train.csv")
dftest=pd.read_csv("../input/test.csv")

dflabel=dftrain.SalePrice
dftrain=dftrain.drop(['Id','SalePrice'],axis=1)
dftest=dftest.drop(['Id'],axis=1)

def normalize_num(df):
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.object:
            df[column] = np.log1p(df[column])
    return df
            


dftrain.shape
dftest.shape
#Handling training set
train_missing_value=dftrain.isnull().sum()
cols_to_drop=train_missing_value[train_missing_value > dftrain.shape[0]/3]
dftrain.drop(list(cols_to_drop.index),axis=1,inplace=True)

dftrain_num_features=dftrain.select_dtypes(exclude=['object'])
dftrain_cat_features=dftrain.select_dtypes(include=['object'])
dftrain_num_features=dftrain_num_features.fillna(dftrain_num_features.median())
cat_to_drop=[cols for cols in dftrain_cat_features.columns 
             if dftrain_cat_features[cols].isnull().any()]
dftrain_cat_features=dftrain_cat_features.drop(cat_to_drop,axis=1)
dftrain_cat_features.isnull().sum()

dftrain=pd.concat((dftrain_num_features,dftrain_cat_features),axis=1)
dftrain.head()

##Hadndle test set
dftest.drop(list(cols_to_drop.index),axis=1,inplace=True)
dftest_num_features=dftest.select_dtypes(exclude=['object'])
dftest_cat_features=dftest.select_dtypes(include=['object'])
dftest_num_features=dftest_num_features.fillna(dftest_num_features.median())
dftest_cat_features=dftest_cat_features.drop(cat_to_drop,axis=1)
dftest=pd.concat((dftest_num_features,dftest_cat_features),axis=1)
dftest.head()

#Get the total length of train set
len_train_set=len(dftrain)
len_train_set

#club train and test dataset
df=pd.concat((dftrain,dftest),axis=0)

from scipy.stats import skew 
skewness = df['BsmtFinSF2'].skew()
skewness

df=normalize_num(df)
skewness = df['BsmtFinSF2'].skew()
skewness
df.head()
#handle categorical features
dfdummy=pd.get_dummies(df)
dfdummy.shape

#disjoin train and test dataset
import copy
dftrain=copy.copy(dfdummy[:len_train_set])
dftest=copy.copy(dfdummy[len_train_set:])
dftrain.shape
dftest.shape

##apply estimator
from sklearn.model_selection import cross_val_score
def rmse_CV_train(model):
    rmse = np.sqrt(-cross_val_score(model,dftrain,dflabel,scoring ="neg_mean_squared_error",cv=20))
    return (rmse)

from sklearn.linear_model import LinearRegression, RidgeCV
lr = LinearRegression()
lr.fit(dftrain,dflabel)
test_pre = lr.predict(dftest)
train_pre = lr.predict(dftrain)
print('rmse on train',rmse_CV_train(lr).mean())

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(dftrain,dflabel)
alpha = ridge.alpha_
print('best alpha',alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)
ridge.fit(dftrain,dflabel)
alpha = ridge.alpha_
print("Best alpha :", alpha)
print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())

y_train_rdg = ridge.predict(dftrain)
y_test_rdg = ridge.predict(dftest)

dftrain.shape



