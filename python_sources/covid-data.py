#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
train_data.head()
train_data['label']='train'


# In[ ]:


test_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
test_data.head()
test_data['label']='score'
concat_data=pd.concat([train_data, test_data])


# In[ ]:


fatal=train_data.loc[train_data.Target == 'Fatalities']["Weight"]
print("rate of fatalities:" ,(sum(fatal)/len(fatal)))


# In[ ]:


cc=train_data.loc[train_data.Target == 'ConfirmedCases']["Weight"]
print("rate of confirmed cases:" ,(sum(cc)/len(cc)))


# from sklearn.ensemble import RandomForestClassifier
# from pandas.api.types import CategoricalDtype
# train_data['Target'] = train_data['Target'].astype('category')
# test_data['Target'] = test_data['Target'].astype(CategoricalDtype(categories=train_data['Target'].cat.categories))
# 
# y = train_data["Target"]
# 
# features = ["Country_Region", "Population", "Weight", "Date"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# 
# model = RandomForestClassifier(n_estimators=234, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)
# 
# output = pd.DataFrame({'ForecastId': test_data.PassengerId, 'Target': predictions})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")
