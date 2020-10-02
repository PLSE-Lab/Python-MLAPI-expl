# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.preprocessing import StandardScaler
import scipy
import sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train =pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
x_train,x_validation,y_train,y_validation=model_selection.train_test_split(train.drop(['id','target'],axis=1),train.target,test_size=0.2,random_state=42)

print(x_train)
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['sub_feature'] = 0.5
params['num_leaves'] = 50
params['min_data'] = 50
clf = lgb.train(params, d_train, 2000 , valid_sets=d_train , early_stopping_rounds=100)
y_pred=clf.predict(x_validation)
roc_auc_score(y_validation,y_pred)
test_pred = clf.predict(test.drop('id',axis=1))
submission['target'] = test_pred
submission.to_csv('submission1.csv',index=False)
    