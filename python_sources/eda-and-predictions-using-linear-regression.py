#!/usr/bin/env python
# coding: utf-8

# # YouTube Channel Data Analysis

# In this kernel we will use data about top 5000 YouTube channels from socialblade to do some basic analysis.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/data.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


p = sns.countplot(data=data, x='Grade')
_ = plt.title('Grade')


# In[ ]:


data['Subscribers'] = pd.to_numeric(data['Subscribers'], errors='coerce')
data['Video Uploads'] = pd.to_numeric(data['Video Uploads'], errors='coerce')
data['Video views'] = pd.to_numeric(data['Video views'], errors='coerce')


# In[ ]:


data_top_20 = data.head(20)


# In[ ]:


sns.set(style='whitegrid')
p = sns.countplot(data=data_top_20, x='Grade')
_ = plt.title('Grade')


# In[ ]:


p = sns.barplot(data=data_top_20, x='Channel name', y='Subscribers')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


p = sns.barplot(data=data_top_20, x='Channel name', y='Video Uploads')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


p = sns.barplot(data=data_top_20, x='Channel name', y='Video views')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


p = sns.scatterplot(data=data_top_20, x='Subscribers', y='Video views', color='red')
_ = plt.title('Subscribers vs Views')


# In[ ]:


p = sns.scatterplot(data=data_top_20, x='Subscribers', y='Video Uploads', color='red')
_ = plt.title('Subscribers vs Views')


# In[ ]:


X = data[['Video Uploads', 'Video views']]
Y = data[['Subscribers']]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
Y = my_imputer.fit_transform(Y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[ ]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


MAE = mean_absolute_error(predictions, y_test)
MSE = mean_squared_error(predictions, y_test)
R2S = r2_score(predictions, y_test)


# In[ ]:


print("Mean Absolute Error", MAE)
print("Mean Squared Error", MSE)
print("R2 Score", R2S)


# In[ ]:




