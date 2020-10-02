# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#importing dataset
train_data = pd.read_csv('../input/train.xls')
test_data = pd.read_csv('../input/test.xls')
sample_submission = pd.read_csv('../input/sample_submission.xls')
#Removing coloms
train_data.drop(['ID','Username'],axis=1,inplace=True)
test_data.drop(['ID','Username'],axis=1,inplace=True)
#Splitting data
X_Train = train_data.iloc[:,:-1].values
X_test = test_data.iloc[:,:].values
y_Train = train_data.iloc[:,-1].values
#Preprocessing the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X_Train[:,0] = le.fit_transform(X_Train[:,0])
X_test[:,0] = le.transform(X_test[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_Train = pd.DataFrame(onehotencoder.fit_transform(X_Train).toarray())
X_test = pd.DataFrame(onehotencoder.transform(X_test).toarray())
X_Train.drop([0],axis=1,inplace=True)
X_test.drop([0],axis=1,inplace=True)
#Train test split
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_Train, y_Train, test_size = 0.4, random_state = 10)
#Making the model
#Model 1 - XGB
import xgboost as xgb
#mod1 = xgb.XGBRegressor(n_estimators=1000, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=1,eval_metric='rmse')
#mod1.fit(X_train,y_train)
#mod1_X_train = mod1.predict(X_train)
#mod1_X_test = mod1.predict(X_val)
mod_hyp1 = xgb.XGBRegressor(n_estimators=500, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=1,eval_metric='rmse')
mod_hyp1.fit(X_Train,y_Train)
ans1=mod_hyp1.predict(X_test)
#Seeing the score
#from sklearn.metrics import mean_squared_error as mse
#err = mse(y_val,mod1_X_test)
#Making Submission file
sample_submission['Upvotes'] = ans1
sample_submission.to_csv("submission_1.csv",index=False)





