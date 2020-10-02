#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Load the Data**

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv',parse_dates = ['Date'])
test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
submission = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')


# # **EDA FOR THE DATASET**

# In[ ]:


train.head()


# In[ ]:


train.info()


# First Convert the Date in Train dataset for the further analysis and you can do this by using the .to_datetime function or you can use parse_date at the time of reading the csv file.as I used at the reading of the file

# In[ ]:


date_data = train['Date']
confirmed_cases = train['ConfirmedCases']
plt.figure(figsize=(10,8))
plt.plot(date_data,confirmed_cases)
plt.xticks(rotation=90)
plt.title('Time Series Analysis for the confirmed Cases')
plt.show()


# In[ ]:


date_data = train['Date']
Fatalities = train['Fatalities']
plt.figure(figsize=(10,8))
plt.plot(date_data,Fatalities)
plt.xticks(rotation=90)
plt.title('Time Series Analysis for Fatalities')
plt.show()


# So, as we can see as we incresed the confrimed_cases gov also incerased the Falaliets

# In[ ]:


train.head()


# In[ ]:


train['Province/State'].value_counts()


# In[ ]:


plt.figure(figsize=(20,5))
sns.countplot(y = train['ConfirmedCases'])
plt.title('Count for confirmed cases')
plt.show()


# In[ ]:


plt.title('Count for confirmed cases')
sns.distplot(train['ConfirmedCases'],kde = False,bins=20)


# We will concentrate only those columns which having confirmed cases as greater than zero

# In[ ]:


train_new = train[train['ConfirmedCases'] > 0]
train_new


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='Date',y='ConfirmedCases',data=train_new)
plt.xticks(rotation=45)
plt.title('Confirmed cases as per Date')
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='Date',y='Fatalities',data=train_new)
plt.xticks(rotation=45)
plt.title('Confirmed Death as per Date')
plt.show()


# # **Feature Engineering**

# **Now let's do some Feature Engineering**

# In[ ]:


train_new.head()


# In[ ]:


train_new['Week'] = train_new['Date'].dt.week
train_new['Day'] = train_new['Date'].dt.day
train_new['DayOfWeek'] = train_new['Date'].dt.dayofweek
train_new['DayOfYear'] = train_new['Date'].dt.dayofyear
train_new.head()


# Now, drop the all other columns as they having the similer data and not going to hamper on our model

# In[ ]:


df = train_new[['Date','Week','Day','DayOfWeek','DayOfYear','ConfirmedCases','Fatalities']]
df.head()


# It's time for using models and fitting our dataset for obtaining the results. We will use few models for the same.

# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,roc_auc_score
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop(['Date','ConfirmedCases','Fatalities'],axis=1)
y = df[['ConfirmedCases','Fatalities']]


# Split the train and test dataset

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


print(f'Size of X_train : {X_train.shape}')
print(f'Size of X_test : {X_test.shape}')
print(f'Size of y_train : {y_train.shape}')
print(f'Size of y_test : {y_test.shape}')


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


def predict_confirmed_cases(regression_algo):
    r = regression_algo()
    r.fit(X_train,y_train['ConfirmedCases'])
    y_pred = r.predict(X_test)
    rSquare = r2_score(y_test['ConfirmedCases'],y_pred)
    confirmed_cases.append(rSquare)

def predict_confirmed_deths(algos):
    r = algos()
    r.fit(X_train,y_train['Fatalities'])
    y_pred = r.predict(X_test)
    rSquare = r2_score(y_test['Fatalities'],y_pred)
    confirmed_death.append(rSquare)
    
models = [KNeighborsRegressor,LinearRegression,RandomForestRegressor,DecisionTreeRegressor,BayesianRidge,
          GradientBoostingRegressor,Lasso]

confirmed_cases = []
confirmed_death = []


# In[ ]:


for i in models:
    predict_confirmed_cases(i)


# In[ ]:


for j in models:
    predict_confirmed_deths(j)


# In[ ]:


confirmed_cases


# In[ ]:


confirmed_death


# In[ ]:


models = pd.DataFrame({
    'Model': ["KNeighborsRegressor","LinearRegression","RandomForestRegressor","DecisionTreeRegressor","BayesianRidge",
          "GradientBoostingRegressor","Lasso"],
    'ConfirmedCase_r2': confirmed_cases,
    'Fatalities_r2' : confirmed_death
})


# In[ ]:


models


# As we can see in the above result by using **KNeighborsRegressor** and **GradientBoostingRegressor** we are getting best result

# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


test_data = test[['ForecastId','Date']]
test_data.head()


# In[ ]:


test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Week'] = test_data['Date'].dt.week
test_data['Day'] = test_data['Date'].dt.day
test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek
test_data['DayOfYear'] = test_data['Date'].dt.dayofyear
test_data.head()


# Create the model to fit on test dataset

# In[ ]:


Kneighbour = KNeighborsRegressor()
Kneighbour.fit(X_train,y_train['ConfirmedCases'])


# In[ ]:


decisiontree = GradientBoostingRegressor()
decisiontree.fit(X_train,y_train['Fatalities'])


# In[ ]:


test_data['ConfirmedCases'] = Kneighbour.predict(test_data.drop(['Date','ForecastId'],axis=1))


# In[ ]:


test_data['Fatalities'] = decisiontree.predict(test_data.drop(['Date','ForecastId','ConfirmedCases'],axis=1))


# In[ ]:


test_data.head()


# In[ ]:


test_data = test_data[['ForecastId','ConfirmedCases','Fatalities']]


# In[ ]:


test_data.head()


# In[ ]:


test_data.to_csv('submission.csv',index=False)


# In[ ]:




