#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 9999


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')


# In[ ]:


train.head(2)


# In[ ]:


submission.head(2)


# In[ ]:


test.head(2)


# In[ ]:


train = train.fillna(-20)


# In[ ]:


test = test.fillna(-20)


# In[ ]:


print('Count of unique regions in test data', len(test['Country/Region'].unique()))
print('Count of unique regions in train data', len(train['Country/Region'].unique()))


# # FE

# train part

# In[ ]:


train['Date'] =  pd.to_datetime(train['Date'])


# In[ ]:


train['year'] = train['Date'].dt.year
train['month'] = train['Date'].dt.month

train['dayofweek'] = train['Date'].dt.dayofweek
train['dayofyear'] = train['Date'].dt.dayofyear
train['weekofyear'] = train['Date'].dt.weekofyear


# In[ ]:


train.head(5)


# test part

# In[ ]:


test['Date'] =  pd.to_datetime(test['Date'])


# In[ ]:


test['year'] = test['Date'].dt.year
test['month'] = test['Date'].dt.month

test['dayofweek'] = test['Date'].dt.dayofweek
test['dayofyear'] = test['Date'].dt.dayofyear
test['weekofyear'] = test['Date'].dt.weekofyear


# In[ ]:


test.head(5)


# # Modelling

# Let's build models for each region

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

ConfirmedCasesList = []
FatalitiesList = []

predictions = []

for region in train['Country/Region'].unique():
    sub_train = train[train['Country/Region'] == region]
    sub_test= test[test['Country/Region'] == region]
    
    sub_clf1 = KNeighborsRegressor(n_neighbors=5).fit(sub_train[['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear']], sub_train[['ConfirmedCases']])
    sub_clf2 = KNeighborsRegressor(n_neighbors=5).fit(sub_train[['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear']], sub_train[['Fatalities']])
    
    ConfirmedCasesList += [x for x in sub_clf1.predict(sub_test[['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear']])]
    FatalitiesList += [x for x in sub_clf2.predict(sub_test[['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear']])]


# In[ ]:


submission.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)
submission['ConfirmedCases'] = ConfirmedCasesList
submission['Fatalities'] = FatalitiesList


# In[ ]:


submission.head(5)


# In[ ]:


submission['ConfirmedCases'] = submission['ConfirmedCases'].str.get(0)
submission['Fatalities'] = submission['Fatalities'].str.get(0)

#submission['ConfirmedCases'] = submission['ConfirmedCases'].apply(np.ceil)
#submission['Fatalities'] = submission['Fatalities'].apply(np.ceil)


# In[ ]:


submission.to_csv('submission.csv', index=False)

