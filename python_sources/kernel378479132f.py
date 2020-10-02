#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import category_encoders as ce
import xgboost as xgb
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import xgboost as xgb
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')


# In[ ]:


x = pd.DataFrame({"x": train['Category'] ,"y" : train['Category'].value_counts()})
x = pd.DataFrame(train['Category'].value_counts()).reset_index()
x.columns = ["Category","val"]


# In[ ]:


train = pd.merge(train,x,on='Category')

train['weight'] = 1/train['val']


# In[ ]:


for df in [train, test]:
    df['Dates']= pd.to_datetime(df['Dates'], format='%Y-%m-%d %H:%M:%S')
    df['Year'] = df['Dates'].apply(lambda x : x.date().year)
    df['Month'] = df['Dates'].apply(lambda x : x.date().month)
    df['Hour'] = df['Dates'].apply(lambda x : x.time().hour)
  
    


# In[ ]:


Y = train.Category

train = train.drop(["Address","Resolution","Descript","Category","Dates"],axis=1)
test = test.drop(["Address","Dates"],axis=1)


# In[ ]:


ce_TE = ce.OneHotEncoder(cols=['PdDistrict'])
ce_TE.fit(train)

train = ce_TE.transform(train)
ce_TE.fit(test)

test = ce_TE.transform(test)


# In[ ]:


ce_E = ce.OrdinalEncoder(cols=['Year','DayOfWeek'])
ce_E.fit(train)

train = ce_E.transform(train)

ce_E.fit(test)

test = ce_E.transform(test)


# In[ ]:


train


# In[ ]:


train['IsDay'] = 0
train.loc[ (train.Hour > 6) & (train.Hour < 20), 'IsDay' ] = 1
test['IsDay'] = 0
test.loc[ (test.Hour > 6) & (test.Hour < 20), 'IsDay' ] = 1


# In[ ]:


ce_L = ce.OrdinalEncoder(cols=['Category'])
ce_L.fit(Y)
Y = ce_L.transform(Y)
Y["Category"] = Y["Category"] - 1


# In[ ]:


train


# In[ ]:


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


encode(train, 'Month', 12)
encode(test, 'Month', 12)

encode(train, 'Hour', 24)
encode(test, 'Hour', 24)


encode(train, 'DayOfWeek', 7)
encode(test, 'DayOfWeek', 24)


# In[ ]:


Y["Category"].value_counts()


# In[ ]:


data_dmatrix = xgb.DMatrix(data=train,label=Y)
xg_reg = xgb.XGBRegressor(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10, num_class=39)


# In[ ]:


xg_reg.fit(train,Y)


# In[ ]:


preds = xg_reg.predict(test.iloc[:,1:])


# In[ ]:


submission = pd.DataFrame(preds)


# In[ ]:


submission.columns = ce_L.mapping[0]['mapping'].index[:-1]


# In[ ]:


submission.insert(0,"Id",test.Id)


# In[ ]:


submission.to_csv("Submission.csv",index=False)


# In[ ]:


# from sklearn.metrics import log_loss


# In[ ]:


# log_loss(Y['Category'],xg_reg.predict(train))


# In[ ]:




