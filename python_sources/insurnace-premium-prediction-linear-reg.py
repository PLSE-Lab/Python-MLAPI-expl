#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/insurance.csv")
df.head()
df.shape


# In[ ]:


df.info()


# In[ ]:


df.sex.value_counts()


# In[ ]:


df.smoker.value_counts()


# In[ ]:


df.region.value_counts()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df[df.duplicated()]


# In[ ]:


df=df.drop_duplicates()


# In[ ]:


df.iloc[575:585,:]


# In[ ]:


import seaborn as sns
sns.pairplot(df)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,3))
corr=df.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


df.head()


# In[ ]:


num_col = df.select_dtypes(include=np.number).columns
cat_col = df.select_dtypes(exclude=np.number).columns


# In[ ]:


num_col


# In[ ]:


cat_col


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in cat_col:
    df[i] = label_encoder.fit_transform(df[i])


# In[ ]:


df.head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
y=df['expenses']
X=df.drop(columns='expenses')
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


model = LinearRegression()
model.fit(train_X,train_y)


# In[ ]:


print("intercept value is*****", model.intercept_)
print("coeffeicint value is*****", model.coef_)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


print("Predicting the train data")
train_predict = model.predict(train_X)
print("Predicting the test data")
test_predict = model.predict(test_X)
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))


# In[ ]:


print("MAPE calculation *************")
def mean_absolute_percentage_error(train_y, train_predict): 
    train_y, train_predict = np.array(train_y), np.array(train_predict)
    return np.mean(np.abs((train_y - train_predict) / train_y)) * 100
mean_absolute_percentage_error(train_y, train_predict)

