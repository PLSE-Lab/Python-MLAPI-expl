#!/usr/bin/env python
# coding: utf-8
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


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/trainandvalid/TrainAndValid.csv')
test = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Test.csv')


# In[ ]:


train.head()


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


train.describe(include='all')


# In[ ]:


train['SalePrice'] = np.log(train.SalePrice)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def model_score(model, X_trn, y_trn, X_val, y_val):
    '''
    Returns the RMSLE Score for the given model
    '''
    model.fit(X_trn, y_trn)
    pred =model.predict(X_val)
    return np.sqrt(mse(pred, y_val))


# In[ ]:


model= RandomForestRegressor()
feature = ['YearMade']


# In[ ]:


X_zero = train[feature]
y_zero = train.SalePrice


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_zero, y_zero, test_size=0.3, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# In[ ]:


model1 = LinearRegression()
model_score(model1, X_trn, y_trn, X_val, y_val)


# In[ ]:


model2 = XGBRegressor()
model_score(model2, X_trn, y_trn, X_val, y_val)


# In[ ]:


#for i in range(50, 500, 50):
#    model3 = XGBRegressor(n_estimators=200)
#    scr = model_score(model3, X_trn, y_trn, X_val, y_val)
#    print(i, '\t', scr)


# In[ ]:


train.datasource.unique()


# In[ ]:


features = ['YearMade', 'datasource']


# In[ ]:


X_one = train[features]
y_one = train.SalePrice


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_one, y_one, test_size=0.3, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


features = ['YearMade', 'datasource', 'state']


# In[ ]:


X_two = train[features]
y_two = train.SalePrice


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


enc = LabelEncoder()
X_two['state']=enc.fit_transform(X_two.state)


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_two, y_two, test_size=0.2, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


train['age']= train.saledate.str[-9:-5].astype(int) - train.YearMade
test['age'] = test.saledate.str[-9:-5].astype(int) - test.YearMade


# In[ ]:


features = ['YearMade', 'datasource', 'state', 'age']
X_three = train[features]
y_three = train.SalePrice


# In[ ]:


X_three['state']= enc.fit_transform(X_three.state)


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_three, y_three, test_size=0.2, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel']
X_four = train[features]
y_four = train.SalePrice


# In[ ]:


X_four['state']= enc.fit_transform(X_four.state)
X_four['fiBaseModel']= enc.fit_transform(X_four.fiBaseModel)


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_four, y_four, test_size=0.2, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


model.score(X_val, y_val)


# In[ ]:


features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel', 'fiProductClassDesc' ]
X_five = train[features]
y_five = train.SalePrice


# In[ ]:


X_five['state']= enc.fit_transform(X_five.state)
X_five['fiBaseModel']= enc.fit_transform(X_five.fiBaseModel)
X_five['fiProductClassDesc']= enc.fit_transform(X_five.fiProductClassDesc)


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_five, y_five, test_size=0.2, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel', 'fiProductClassDesc' , 'fiModelDesc']
X_six = train[features]
y_six = train.SalePrice


# In[ ]:


X_test = test[features]


# In[ ]:


net_state=X_six.state
net_state= net_state.append(X_test.state, ignore_index=True)


# In[ ]:


net_state


# In[ ]:


enc_st = LabelEncoder()
enc_st.fit(net_state)


# In[ ]:


X_six['state']= enc_st.transform(X_six.state)


# In[ ]:


X_test['state'] = enc_st.transform(X_test.state)


# In[ ]:


net_pcd = X_six.fiProductClassDesc
net_pcd = net_pcd.append(X_test.fiProductClassDesc, ignore_index=True)
net_pcd


# In[ ]:


enc_pcd = LabelEncoder()
enc_pcd.fit(net_pcd)


# In[ ]:


X_six.fiProductClassDesc = enc_pcd.transform(X_six.fiProductClassDesc)


# In[ ]:


X_test['fiProductClassDesc'] = enc_pcd.transform(X_test.fiProductClassDesc )


# In[ ]:


net_bm = X_six.fiBaseModel
net_bm = net_bm.append(X_test.fiBaseModel, ignore_index=True)
net_bm


# In[ ]:


enc_bm = LabelEncoder()
enc_bm.fit(net_bm)


# In[ ]:


X_six.fiBaseModel = enc_bm.transform(X_six.fiBaseModel)


# In[ ]:


X_test.fiBaseModel = enc_bm.transform(X_test.fiBaseModel)


# In[ ]:


X_test


# In[ ]:


net_md = X_six.fiModelDesc
net_md = net_md.append(X_test.fiModelDesc, ignore_index=True)
net_md


# In[ ]:


enc_md = LabelEncoder()
enc_md.fit(net_md)


# In[ ]:


X_six['fiModelDesc'] = enc_md.transform(X_six.fiModelDesc)


# In[ ]:


X_test['fiModelDesc'] = enc_md.transform(X_test.fiModelDesc)


# In[ ]:


X_six


# In[ ]:


X_trn, X_val, y_trn, y_val = train_test_split(X_six, y_six, test_size=0.2, random_state=0)


# In[ ]:


model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


model.score(X_val, y_val)


# In[ ]:


#for i in range(10,250, 10):
#    model = RandomForestRegressor(n_estimators= i)
#    print(i, '\t',  model_score(model, X_trn, y_trn, X_val, y_val))


# In[ ]:


model = RandomForestRegressor(n_estimators=110, n_jobs= -1)
model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


model.score(X_val, y_val)


# In[ ]:


X_six.describe()


# In[ ]:


model = RandomForestRegressor(max_depth=30, min_samples_split=20, n_estimators=110, n_jobs= -1)
model_score(model, X_trn, y_trn, X_val, y_val)


# In[ ]:


model.score(X_val, y_val)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:


train.Engine_Horsepower.unique()


# In[ ]:


train.fiBaseModel


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train.state.unique()


# In[ ]:


test.state.unique()


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': test.SalesID,
                       'SalePrice': np.exp(pred)})
output.to_csv('submission.csv', index=False)


# In[ ]:




