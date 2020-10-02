#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe().transpose()


# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(df["price"])


# In[ ]:


sns.countplot(df["bedrooms"])


# In[ ]:


df.corr()["price"].sort_values()


# In[ ]:


plt.figure(figsize=(10,5))
sns.scatterplot(x="price",y="sqft_living",data=df)


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="bedrooms",y="price",data=df)


# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="price",y="long",data=df,color="pink")


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="price",y="lat",data=df,color="yellow")


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="long",y="lat",data=df,color="green",hue="price")


# In[ ]:


df.sort_values("price",ascending=False).head(20)


# In[ ]:


len(df)*0.01


# In[ ]:


#getting away from top 1%
non_top_1_prec =df.sort_values("price",ascending=False).iloc[216:]


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="long",y="lat",data=non_top_1_prec,
                edgecolor=None,alpha=0.2,palette="RdYlGn",hue="price")


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="waterfront",y="price",data=df)


# In[ ]:


df=df.drop("id",axis=1)


# In[ ]:


df["date"]=pd.to_datetime(df["date"])


# In[ ]:


df["year"]=df["date"].apply(lambda date:date.year)
df["month"]=df["date"].apply(lambda date:date.month)


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="month",y="price",data=df)


# In[ ]:


df.groupby("year").mean()["price"].plot()


# In[ ]:


df=df.drop("date",axis=1)


# In[ ]:


df.head()


# In[ ]:


df["zipcode"].value_counts()


# In[ ]:


#not of much use

df=df.drop("zipcode",axis=1)


# In[ ]:


df["yr_renovated"].value_counts()


# In[ ]:


X=df.drop("price",axis=1)
y=df["price"].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=25)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


model=Sequential()
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")


# In[ ]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[ ]:


predictions=model.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test,predictions))


# In[ ]:


df["price"].describe()


# In[ ]:


explained_variance_score(y_test,predictions)


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,"r")


# In[ ]:




