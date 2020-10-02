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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')


# In[ ]:


df


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(df['price'])


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe().transpose()


# In[ ]:


sns.countplot(df['bedrooms'])


# In[ ]:


df.corr()['price'].sort_values()


# In[ ]:


plt.figure(figsize=(12,5))
sns.scatterplot('price', 'sqft_living', data=df)


# In[ ]:


sns.scatterplot('price', 'lat', data=df)


# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot('long', 'lat', data=df, hue='price')


# In[ ]:


outlier_top_one = df.sort_values('price', ascending=False).iloc[216:]


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(outlier_top_one['price'])


# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot('long', 'lat', data=outlier_top_one, 
                hue='price')


# In[ ]:


sns.boxplot('waterfront', 'price', data=df)


# In[ ]:


df = df.drop('id', axis=1)


# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot('month', 'price', data=df)


# In[ ]:


df = df.drop('date', axis=1)


# In[ ]:


df.groupby('month').mean()['price'].plot()


# In[ ]:


df = df.drop('zipcode', axis=1)


# In[ ]:


X = df.drop('price', axis=1).values


# In[ ]:


y = df['price'].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
         batch_size=128,
         epochs=444)


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, explained_variance_score, balanced_accuracy_score 
predictions = model.predict(X_test)
mean_squared_error(y_test, predictions)**0.5
mean_absolute_error(y_test, predictions)
explained_variance_score(y_test, predictions)


# In[ ]:


plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'y')


# In[ ]:


single_first_house = df.drop('price', axis=1).iloc[0]
single_first_house = scaler.transform(single_first_house.values.reshape(-1, 19))
model.predict(single_first_house)
df.head(1)


# In[ ]:




