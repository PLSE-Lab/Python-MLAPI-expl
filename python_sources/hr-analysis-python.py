# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Read Data
data_hr = pd.read_csv("../input/HR_comma_sep.csv")
print(data_hr[:3])


data_hr['salary'].replace({'low':1,'medium':5,'high':10},inplace = True)
data_hr['salary'].unique()

dummies=pd.get_dummies(data_hr['sales'],prefix='sales')
data_hr=pd.concat([data_hr,dummies],axis=1)
data_hr.drop(['sales'],axis=1,inplace=True)
data_hr.head(10)


#Split Data 
split_len = np.random.rand(len(data_hr)) <0.8
data_train = data_hr[split_len]
data_test  = data_hr[~split_len]
len(data_test)
len(data_train)
data_test[:0]

#Random Forest
model=RandomForestRegressor(n_estimators=100,n_jobs=-1,oob_score=True,random_state=19)
train_x = data_train.drop(['left'],axis = 1)
train_y = data_train['left']

test_x = data_test.drop(['left'],axis = 1)
test_y = data_test['left']


model.fit(train_x,train_y)
model_score = model.score(test_x,test_y)
print (model_score)