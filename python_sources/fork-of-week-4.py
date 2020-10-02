#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

pd.set_option('display.max_rows', 500)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


df_train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")


# In[ ]:


df_submission2 = pd.read_csv("../input/submission2/submission_y.csv")


# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])


# In[ ]:


df_train.info()


# In[ ]:


df_train['Country_Region'].value_counts()
df_test['Country_Region'].value_counts()


# In[ ]:


# To find total number of countries in train and test
df_train['Country_Region'].nunique()
len(df_test['Country_Region'].unique())


# In[ ]:


# To see count of record for each date in train and test
print("Train")
df_train.groupby([df_train['Date']])['Country_Region'].count()
print("Test")
df_test.groupby([df_test['Date']])['Country_Region'].count()


# In[ ]:


from pandas_profiling import ProfileReport
train_profile = ProfileReport(df_train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile


# In[ ]:


# there are common dates between train and test data so we need to remove those dates data from df_train
max_test_dt = df_test['Date'].min()
df_train = df_train[df_train['Date'] < max_test_dt]


# In[ ]:


df_train.tail()


# In[ ]:


# Sorted based on Confirmed Cases
df_train.groupby(by='Country_Region')['ConfirmedCases'].agg(np.max).sort_values(ascending=False).plot()


# In[ ]:


# Sorted based on Fatalities
df_train.groupby(by='Country_Region')['Fatalities'].agg(np.max).sort_values(ascending=False).plot()


# In[ ]:


# Lets Analyze China
# Create figure and plot space
# fig, ax = plt.subplots(figsize=(10, 10))

# # Add x-axis and y-axis
# ax.plot(df_train[df_train['Country_Region'] == 'China']['Date'],
#         df_train[df_train['Country_Region'] == 'China']['ConfirmedCases'],
#         color='purple')
# ax.plot(df_train[df_train['Country_Region'] == 'China']['Date'],
#         df_train[df_train['Country_Region'] == 'China']['Fatalities'],
#         color='red')


# # Set title and labels for axes
# ax.set(xlabel="Date",
#        ylabel="ConfirmedCases",
#        title="China Confirmed and Fatalities")

# plt.show()

# df_train[df_train['Country_Region'] == 'China'].plot(x="Date", y=["ConfirmedCases", "Fatalities"],logy=True)
# plt.show()


df_train[df_train['Country_Region'] == 'China'][["ConfirmedCases", "Fatalities"]].plot()
plt.show()


# In[ ]:


df_train[df_train['Country_Region'] == 'United Kingdom'][["ConfirmedCases", "Fatalities"]].plot()
plt.show()


# In[ ]:


df_train.head()


# In[ ]:


df_train['Province_State'] = df_train['Province_State'].fillna(value="None")
df_test['Province_State'] = df_test['Province_State'].fillna(value="None")


# In[ ]:


df_train.head()


# In[ ]:


df_train['Country_Region'] = np.where(df_train['Province_State'].str.contains('None'), 
                             df_train['Country_Region'],df_train['Country_Region'] + '_' + df_train['Province_State'])


# In[ ]:


df_test['Country_Region'] = np.where(df_test['Province_State'].str.contains('None'), 
                             df_test['Country_Region'],df_test['Country_Region'] + '_' + df_test['Province_State'])


# In[ ]:


df_train.head()


# In[ ]:


df_train['dayofyear'] = df_train['Date'].dt.dayofyear
df_train['month'] = df_train['Date'].dt.month

df_test['dayofyear'] = df_test['Date'].dt.dayofyear
df_test['month'] = df_test['Date'].dt.month


# In[ ]:


df_train.head()


# In[ ]:


le = preprocessing.LabelEncoder()
df_train['Country_Region'] = le.fit_transform(df_train['Country_Region'])
df_test['Country_Region'] = le.transform(df_test['Country_Region'])


# In[ ]:


df_train.drop(['Id', 'Date','Province_State'], axis=1,inplace=True)
df_test.drop(['ForecastId', 'Date','Province_State'], axis=1,inplace=True)


# In[ ]:


def calc_score(y_true, y_pred):
    y_true[y_true<0] = 0
    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5
    return score


# In[ ]:


# from sklearn.ensemble import RandomForestRegressor
# Tree_model = RandomForestRegressor(max_depth=200, random_state=0)


# In[ ]:


from xgboost import XGBRegressor
model_xgb = XGBRegressor(n_estimators=1500)


# In[ ]:


df_train.head()


# In[ ]:


col = ['Country_Region','dayofyear','month']
y1 = ['ConfirmedCases']
y2 = ['Fatalities']


# In[ ]:


##
model_xgb.fit(df_train[col],df_train[y1])
conf_pred = model_xgb.predict(df_test)


# In[ ]:


model_xgb.fit(df_train[col],df_train[y2])
Fat_pred = model_xgb.predict(df_test)


# In[ ]:


df_submission['ConfirmedCases'] = pd.DataFrame(conf_pred)
df_submission['Fatalities'] = pd.DataFrame(Fat_pred)


# In[ ]:


df_submission.head()
df_submission2.head()


# In[ ]:


avg_con = (df_submission['ConfirmedCases'] + df_submission2['ConfirmedCases'])/2
avg_fat = (df_submission['Fatalities'] + df_submission2['Fatalities'])/2


# In[ ]:


df_submission['ConfirmedCases'] = avg_con
df_submission['Fatalities'] = avg_fat


# In[ ]:


df_submission.to_csv('submission.csv', index=False)


# In[ ]:




