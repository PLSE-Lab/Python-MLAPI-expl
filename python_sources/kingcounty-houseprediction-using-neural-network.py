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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(df['price'])


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(df['bedrooms'])


# In[ ]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)


# In[ ]:


get_ipython().system('pip install folium')


# In[ ]:


import folium


# In[ ]:


world_map=folium.Map()


# In[ ]:


ofg=folium.map.FeatureGroup()
for lat,long in zip(df['lat'],df['long']):
  ofg.add_child(folium.CircleMarker([lat,long],
                                             radius=5,
                                             colour='red',
                                             fill_color='red'))
world_map.add_child(ofg) 


# In[ ]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='price',data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='lat',y='price',data=df)


# In[ ]:


plt.figure(figsize=(20,20))
sns.scatterplot(x='long',y='lat',data=df,hue='price',palette='coolwarm')


# In[ ]:


df.sort_values('price',ascending=False).head(30)


# # **we see that our price range mostly from 1 to 3 lacs.So we can remove the outliers from our data**

# **Removing outliers**

# In[ ]:


len(df)


# In[ ]:


len(df)*0.01


# # We will remove top 1%

# In[ ]:


non_top1_percent=df.sort_values('price',ascending=False).iloc[216:]


# In[ ]:


non_top1_percent


# In[ ]:


plt.figure(figsize=(20,20))
sns.scatterplot(x='long',y='lat',data=non_top1_percent,edgecolor=None,alpha=0.2,hue='price',palette='RdYlGn')


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='waterfront',y='price',data=non_top1_percent)


# # Feature Engineering

# In[ ]:


non_top1_percent.drop('id',axis=1,inplace=True)


# In[ ]:


non_top1_percent['date']=pd.to_datetime(non_top1_percent['date'])


# In[ ]:


non_top1_percent


# In[ ]:


non_top1_percent['year']=non_top1_percent['date'].apply(lambda x:x.year)
non_top1_percent['month']=non_top1_percent['date'].apply(lambda x:x.month)


# In[ ]:


non_top1_percent.head()


# In[ ]:


non_top1_percent.corr()['price'].sort_values(ascending=False)


# In[ ]:


non_top1_percent['current_year']=2020
non_top1_percent.head()


# In[ ]:


non_top1_percent['years_old']=non_top1_percent['current_year']-non_top1_percent['year']
non_top1_percent.head()


# In[ ]:


non_top1_percent.corr()['price'].sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='month',y='price',data=non_top1_percent)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='year',y='price',data=non_top1_percent)


# In[ ]:


non_top1_percent.groupby('month').mean()['price']


# In[ ]:


non_top1_percent.groupby('year').mean()['price']


# In[ ]:


plt.figure(figsize=(12,8))
non_top1_percent.groupby('month').mean()['price'].plot(color='red')


# In[ ]:


plt.figure(figsize=(12,8))
non_top1_percent.groupby('year').mean()['price'].plot(color='red')


# In[ ]:


plt.figure(figsize=(12,8))
non_top1_percent.groupby('years_old').mean()['price'].plot(color='red')


# In[ ]:


non_top1_percent.drop(['date','current_year'],axis=1,inplace=True)


# In[ ]:


non_top1_percent.head()


# In[ ]:


non_top1_percent['zipcode'].value_counts()


# In[ ]:


non_top1_percent['yr_renovated'].value_counts()


# In[ ]:


non_top1_percent.drop('zipcode',axis=1,inplace=True)


# In[ ]:


def yr_renovated_func(x):
  if (x==0):
    non_top1_percent['not_renovated']=0
  else:
    non_top1_percent['renovated']=1
non_top1_percent['yr_renovated'].apply(yr_renovated_func)


# In[ ]:


non_top1_percent['yr_renovated'].value_counts()


# In[ ]:


non_top1_percent.head()


# In[ ]:


non_top1_percent.columns


# In[ ]:


non_top1_percent.drop('yr_renovated',axis=1,inplace=True)


# In[ ]:


non_top1_percent.head()


# In[ ]:


# 


# # **Model Training**

# In[ ]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


X=non_top1_percent.drop('price',axis=1).values
y=non_top1_percent['price'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print('training size',X_train.shape,y_train.shape)
print('test size',X_test.shape,y_test.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


model=Sequential()

model.add(Dense(21,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(21,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[ ]:


model.fit(x=X_train,y=y_train,epochs=500,validation_data=(X_test,y_test),batch_size=128)


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses.head()


# In[ ]:


plt.figure(figsize=(20,8))
losses.plot()


# In[ ]:


print(X_test.shape)
predict=model.predict(X_test)


# In[ ]:


predict


# In[ ]:


model.evaluate(X_test,y_test)


# # **Evaluation And Prediction**

# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score


# In[ ]:


print(' mean absolute error',mean_absolute_error(y_test,predict))
print(' mean squared error',mean_squared_error(y_test,predict))
print('explained_variance_score',explained_variance_score(y_test,predict))


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(y_test,predict)
plt.plot(y_test,y_test,'-r')


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save('kingcountry_houseprice.h5')


# In[ ]:




