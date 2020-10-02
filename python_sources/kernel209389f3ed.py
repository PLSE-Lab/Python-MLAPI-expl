# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/train.csv")
xtrain=dataset.iloc[:,1:80]
ytrain=dataset.iloc[:,80:]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,Imputer
encoder=LabelEncoder()
for i in range(0,xtrain.shape[1]):
    xtrain[xtrain.columns[i]]=encoder.fit_transform(xtrain[xtrain.columns[i]].astype(str))

encoder1=LabelEncoder()
ytrain=encoder1.fit_transform(ytrain)

from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(xtrain,ytrain)

dataset1=pd.read_csv("../input/test.csv")
xtest=dataset1.iloc[:,1:]
for i in range(0,xtrain.shape[1]):
    xtest[xtest.columns[i]]=encoder.fit_transform(xtest[xtest.columns[i]].astype(str))
    
ypred=xgb.predict(xtest)
Y=np.zeros((1459,2))
for i in range(0,xtest.shape[0]):
    Y[i][0]=1461+i
    Y[i][1]=ypred[i]
pd.DataFrame(Y).to_csv("submission.csv",index=None)
print("complete")

ypred=xgb.predict(xtest)

Y=np.zeros((1459,2))

for i in range(0,xtest.shape[0]):
    Y[i][0]=1461+i
    Y[i][1]=ypred[i]
pd.DataFrame(Y).to_csv("submission.csv",index=None)