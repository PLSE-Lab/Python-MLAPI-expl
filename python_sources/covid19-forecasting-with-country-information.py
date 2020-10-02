#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


cleaned_train = pd.read_csv('../input/covid-clean/clean_train.csv')
cleaned_test = pd.read_csv('../input/covid-clean/clean_test.csv')
submission = pd.read_csv('../input/covid-clean/submission.csv')


# In[ ]:


cleaned_train.columns


# In[ ]:


X = cleaned_train[['day_from_jan_first','Lat','Long',
                   'medianage','urbanpop','hospibed',
                   'lung','avgtemp','avghumidity','days_from_firstcase']]


# In[ ]:


X_fat = cleaned_train[['day_from_jan_first','Lat','Long',
                   'medianage','urbanpop','hospibed',
                   'lung','avgtemp','avghumidity','days_from_firstcase','ConfirmedCases']]


# In[ ]:


y1 = cleaned_train[['ConfirmedCases']]
y2 = cleaned_train[['Fatalities']]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor_1=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)
regressor_2=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)
regressor_1.fit(X,y1)
regressor_2.fit(X_fat,y2)


# In[ ]:


test_X = cleaned_test[['day_from_jan_first','Lat','Long',
                   'medianage','urbanpop','hospibed',
                   'lung','avgtemp','avghumidity','days_from_firstcase']]
test_X_fat  = cleaned_test[['day_from_jan_first','Lat','Long',
                   'medianage','urbanpop','hospibed',
                   'lung','avgtemp','avghumidity','days_from_firstcase']]


# In[ ]:


y_conf = regressor_1.predict(test_X)
y_conf = np.where(y_conf<0,0,np.rint(y_conf))


# In[ ]:


test_X_fat['ConfirmedCases'] = y_conf


# In[ ]:


y_fat = regressor_2.predict(test_X_fat)
y_fat = np.where(y_fat<0,0,np.rint(y_fat))


# In[ ]:


submission=pd.DataFrame(columns=submission.columns)


# In[ ]:


submission['ForecastId'] = cleaned_test['ForecastId']
submission['ConfirmedCases'] = y_conf
submission['Fatalities'] = y_fat


# In[ ]:


submission.to_csv('submission.csv',index=False)

