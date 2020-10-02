#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error

import seaborn as sns
import missingno as msno
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score 
from datetime import datetime, timedelta
import os


# ### Import data
# Import train and test data. Join train data and labels. Make a full data set (df) with train and test

# In[ ]:


features_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')
labels_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv')
df_train=pd.merge(features_train, labels_train, on=["city","year","weekofyear"])
df_test=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')
df_test['total_cases'] = 0
df = pd.concat([df_train,df_test])
df.head()


# In[ ]:


feature_names = df.columns[4:]
print(feature_names)


# #### Fill null data copying previous data in series

# In[ ]:


df = df.fillna(method='ffill')


# #### Counts the number of occourences for each city.
# Will threat each city as separated problem

# In[ ]:


features_train.city.value_counts()


# In[ ]:


sj_count = 936
iq_count = 520


# ### Train two models using VAR (time series multi variable)
# for every predicted values, retrain the model and predict next value to test set.

# In[ ]:


data = df[df.city=='sj'][feature_names].reset_index(drop=True)
for i in range(sj_count,len(data)-1):
    model = VAR(endog=data[:i])
    model_fit = model.fit()

    # make prediction on validation
    prediction = model_fit.forecast(model_fit.y, steps=1)
    cases = prediction[0,-1]
    data.at[i+1,'total_cases'] = cases
test_sj_preds = data[sj_count:]['total_cases'].clip_lower(0)


# In[ ]:


data = df[df.city=='iq'][feature_names].reset_index(drop=True)
for i in range(iq_count,len(data)-1):
    model = VAR(endog=data[:i])
    model_fit = model.fit()

    # make prediction on validation
    prediction = model_fit.forecast(model_fit.y, steps=1)
    cases = prediction[0,-1]
    data.at[i+1,'total_cases'] = cases
test_iq_preds = data[iq_count:]['total_cases'].clip_lower(0)


# In[ ]:


df_test.loc[df_test.city=='sj','total_cases'] = test_sj_preds.values
df_test.loc[df_test.city=='iq','total_cases'] = test_iq_preds.values


# In[ ]:


Submission_Deng_AI = df_test[['city','year','weekofyear', 'total_cases']]
Submission_Deng_AI.to_csv("Submission_Deng_AI.csv", index=False)


# ### Plotting the results

# In[ ]:


figure(figsize=(20,10))
plt.plot( 'week_start_date', 'total_cases', data=df_train[df_train.city=='sj'],  
         color='blue', label='sj')
plt.plot( 'week_start_date', 'total_cases', data=df_train[df_train.city=='iq'], color='olive', label='iq')
plt.plot( 'week_start_date', 'total_cases', data=df_test[df_test.city=='sj'],  color='red', linestyle='dashed', 
         label="predictet sj")
plt.plot( 'week_start_date', 'total_cases', data=df_test[df_test.city=='iq'],  color='green',  linestyle='dashed', 
         label="predictet iq")
plt.legend()


# In[ ]:




