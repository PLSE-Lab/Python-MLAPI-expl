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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')


# In[ ]:


df_test=pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')
df_test


# In[ ]:





# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train['Airline'].nunique()


# In[ ]:


df_train['Additional_Info'].unique()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


df_train['Date_of_Journey']=pd.to_datetime(df_train['Date_of_Journey'])
df_train['Day_of_Journey']=(df_train['Date_of_Journey']).dt.day
df_train['Month_of_Journey']=(df_train['Date_of_Journey']).dt.month


# In[ ]:


df_train.head()


# In[ ]:


#Dep_time 
df_train['Dep_hr']=pd.to_datetime(df_train['Dep_Time']).dt.hour
df_train['Dep_min']=pd.to_datetime(df_train['Dep_Time']).dt.minute


# In[ ]:


#Arrival_time
df_train['Arrival_hr']=pd.to_datetime(df_train['Arrival_Time']).dt.hour
df_train['Arrival_min']=pd.to_datetime(df_train['Arrival_Time']).dt.minute


# In[ ]:


#Splitting duration  time

a=df_train['Duration'].str.split(' ',expand=True)
a[1].fillna('00m',inplace=True)
df_train['dur_hr']=a[0].apply(lambda x: x[:-1])
df_train['dur_min']=a[1].apply(lambda x: x[:-1])


# In[ ]:


df_train.head()


# In[ ]:


#dropping the data
df_train.drop(['Date_of_Journey','Duration','Arrival_Time','Dep_Time'],inplace=True,axis=1)


# In[ ]:


df_train.head()


# In[ ]:


air_price=df_train.groupby('Airline')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(15,6))
sns.barplot(air_price.index,air_price.values)
plt.xticks(rotation=90)
#Jet airways have higher  mean prices
#Trujetand spice jet with  the lowest, maybe its because the number of flights operating, let us check


# In[ ]:


#No of flights operating

p=df_train['Airline'].value_counts()
p


# In[ ]:


plt.figure(figsize=(8,8))
plt.pie(p.values, labels=p.index,autopct='%1.1f%%')
#jet flights has more flights operating,  followed by indigo,airindia,and trujet,jetairways,multiple carriers premium economy has very less flights operaring


# In[ ]:


df_train[(df_train['Source']=='Banglore') & (df_train['Destination']=='New Delhi')]


# In[ ]:


#price based on number of stops

df_train.groupby(['Airline','Total_Stops'])['Price'].mean()


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(y=df_train['Airline'],x=df_train['Price'],hue=df_train['Total_Stops'])


# In[ ]:


df_train['Total_Stops'].unique()


# In[ ]:


#Handling Categorical Values 
df_train['Total_Stops']=df_train['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})
dum_airine=pd.get_dummies(df_train['Airline'],drop_first=True)
print(df_train['Source'].value_counts())
print(df_train['Destination'].value_counts())

dum_src_dest=pd.get_dummies(df_train[['Source','Destination']],drop_first=True)
df_train=pd.concat([dum_airine,dum_src_dest,df_train],axis=1)
df_train.drop(['Airline','Source','Destination'],inplace=True,axis=1)


# In[ ]:





# In[ ]:


#TEST DATA


# In[ ]:


df_test['Date_of_Journey']=pd.to_datetime(df_test['Date_of_Journey'])
df_test['Day_of_Journey']=(df_test['Date_of_Journey']).dt.day
df_test['Month_of_Journey']=(df_test['Date_of_Journey']).dt.month

#Dep_time 
df_test['Dep_hr']=pd.to_datetime(df_test['Dep_Time']).dt.hour
df_test['Dep_min']=pd.to_datetime(df_test['Dep_Time']).dt.minute

#Arrival_time
df_test['Arrival_hr']=pd.to_datetime(df_test['Arrival_Time']).dt.hour
df_test['Arrival_min']=pd.to_datetime(df_test['Arrival_Time']).dt.minute

#Splitting duration  time

a=df_test['Duration'].str.split(' ',expand=True)
a[1].fillna('00m',inplace=True)
df_test['dur_hr']=a[0].apply(lambda x: x[:-1])
df_test['dur_min']=a[1].apply(lambda x: x[:-1])

#dropping the data
df_test.drop(['Date_of_Journey','Duration','Arrival_Time','Dep_Time'],inplace=True,axis=1)

#Handling Categorical Values 
df_test['Total_Stops']=df_test['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})

dum_airine=pd.get_dummies(df_test['Airline'],drop_first=True)
dum_src_dest=pd.get_dummies(df_test[['Source','Destination']],drop_first=True)


# In[ ]:


df_test=pd.concat([dum_airine,dum_src_dest,df_test],axis=1)
df_test.drop(['Airline','Source','Destination','Additional_Info',"Route"],inplace=True,axis=1)


# In[ ]:


print('train_shape',df_train.shape)
print('test_shape',df_test.shape)


# In[ ]:


df_train.columns


# In[ ]:


x=df_train.loc[:,['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
       'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Vistara', 'Vistara Premium economy', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi', 'Total_Stops',  'Day_of_Journey',
       'Month_of_Journey', 'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min',
       'dur_hr', 'dur_min']]
y=df_train['Price']


# In[ ]:


plt.figure(figsize=(25,10))
sns.heatmap(df_train.corr(),cmap='RdYlGn',annot=True)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
ext=ExtraTreesRegressor()
ext.fit(x,y)


# In[ ]:


pd.Series(ext.feature_importances_,index=x.columns).sort_values(ascending=False).plot(kind='bar',figsize=(10,10))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 22)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
pred=rf.predict(x_test)


# In[ ]:


sns.scatterplot(pred,y_test)


# In[ ]:


from sklearn.metrics import r2_score
sns.distplot(y_test-pred)
print(r2_score(y_test,pred))


# In[ ]:


#hypeparametre tuning

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 40, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


rand_grid={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_rs=RandomForestRegressor()

rv=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',n_iter=10,cv=3,random_state=42, n_jobs = 1)


# In[ ]:


rv.fit(x_train,y_train)


# In[ ]:


rv.best_params_


# In[ ]:


pred=rv.predict(x_test)
sns.distplot(y_test-pred)
print(r2_score(y_test,pred))


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE',mean_absolute_error(y_test,pred))
print('MSE',mean_squared_error(y_test,pred))


# In[ ]:


sns.scatterplot(y_test,pred)


# In[ ]:


import xgboost as xgb


# In[ ]:


xg=xgb.XGBRegressor()


# In[ ]:


x_train[['dur_hr','dur_min']]=x_train[['dur_hr','dur_min']].astype(int)
x_test[['dur_hr','dur_min']]=x_test[['dur_hr','dur_min']].astype(int)


# In[ ]:


xg.fit(x_train,y_train)


# In[ ]:


pred=xg.predict(x_test)


# In[ ]:


print(r2_score(y_test,pred))
sns.distplot(y_test-pred)
sns.scatterplot(y_test,pred)
print('MAE',mean_absolute_error(y_test,pred))
print('MSE',mean_squared_error(y_test,pred))

