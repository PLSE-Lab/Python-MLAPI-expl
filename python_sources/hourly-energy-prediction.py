#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/PJME_hourly.csv', index_col=[0], parse_dates=[0])


# In[ ]:


df.plot(figsize=(15,8));


# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df.index,
                    y = df.PJME_MW)

data = [trace1]
   
fig = dict(data = data)
iplot(fig)


# **Split the dataset in training and testing set **

# In[ ]:


splitdate = '2014-01-01'
df_train_set = df[df.index < splitdate]
df_test_set = df[df.index > splitdate]


# In[ ]:


df_test_set.head(2)


# In[ ]:


# Creating trace1
trace1 = go.Scatter( x = df_train_set.index, y = df_train_set.PJME_MW)

data = [trace1]
   
fig = dict(data = data)
iplot(fig)


# In[ ]:


trace1 = go.Scatter( x = df_test_set.index, y = df_test_set.PJME_MW)

data = [trace1]
   
fig = dict(data = data)
iplot(fig)


# In[ ]:


def AddDateProperties (df) :
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    return df


# In[ ]:


df_test_set = AddDateProperties(df_test_set)
df_train_set = AddDateProperties(df_train_set)
#Remove Date column for creating Regressor Model
df_test_set = df_test_set.drop(['date'], axis=1)
df_train_set = df_train_set.drop(['date'], axis=1)
y_test = df_test_set['PJME_MW']
y_train = df_train_set['PJME_MW']
X_test = df_test_set.loc[:,df_test_set.columns !='PJME_MW']
X_train = df_train_set.loc[:,df_train_set.columns !='PJME_MW']


# In[ ]:


model = xgb.XGBRegressor( learning_rate= 0.01 , n_estimators=1000 , max_depth=3 , subsample= 0.8 , colsample_bylevel= 1)


# In[ ]:


eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc","error"]


# In[ ]:


model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) 


# In[ ]:


plot_importance(model) ;


# In[ ]:


df_test_set['MW_Prediction'] = model.predict(X_test)


# In[ ]:


df_all = pd.concat([df_test_set,df_train_set] , sort=False)


# In[ ]:


df_all[['PJME_MW' , 'MW_Prediction']].plot(figsize =(15,5));


# # First 1 month trend

# In[ ]:


start_date ='2014-01-01'
end_date = '2014-01-31'
fig,ax = plt.subplots(1)

df_all[['PJME_MW' , 'MW_Prediction']].plot(ax=ax , figsize = (15,5 ) , style = ['.']);
ax.set_xbound(lower=start_date , upper= end_date)


# In[ ]:


df_test_set.head()


# In[ ]:


df_test_set['Error'] = df_test_set['PJME_MW'] - df_test_set['MW_Prediction']


# In[ ]:


df_test_set['AbsError'] = df_test_set.Error.apply(np.abs)


# In[ ]:


day_groupby = df_test_set.groupby(['year','month','dayofmonth'])
error_by_day = day_groupby['PJME_MW','MW_Prediction','Error','AbsError'].mean()


# # Best Predicted Day

# In[ ]:


error_by_day.sort_values(ascending= True , by = 'AbsError').head(15)


# # Worst Predicted day

# In[ ]:


error_by_day.sort_values(ascending= False , by = 'AbsError').head(15)


# In[ ]:


start_date ='2015-02-20 00:00:00'
end_date = '2015-02-20 23:00:00'

fig,ax = plt.subplots(1)

df_all[['PJME_MW' , 'MW_Prediction']].plot(ax=ax , figsize = (15,5 ) , style = ['.']);
ax.set_xbound(lower=start_date , upper= end_date)


# # Predictions where prediction was over the consumptions 

# In[ ]:


error_by_day.sort_values(ascending= True , by = 'Error').head(15)


# In[ ]:


# plot single tree
plot_tree(model,num_trees=1 ,rankdir= 'LR')
plt.show()
plt.rcParams['figure.figsize'] = (100, 70)


# In[ ]:




