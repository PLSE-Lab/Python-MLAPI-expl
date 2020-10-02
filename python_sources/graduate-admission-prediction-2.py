#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualisation
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


df.head()


# In[ ]:


df.set_index('Serial No.',inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


sns.countplot(df['University Rating'])
plt.show()


# In[ ]:


sns.countplot(df['Research'])
plt.show()


# In[ ]:


# Spliting dataframe into input features and target variable
x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


# Splitting Data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)


# In[ ]:


# Training of data using machine learning algorithm
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


y_predict = np.round(lr.predict(x_test),2)


# In[ ]:


r2Score = r2_score(y_test,y_predict)


# In[ ]:


rmse = np.sqrt(mean_squared_error(y_test,y_predict))


# In[ ]:


print("R2 Score of the model is {}".format(r2Score))
print("Root Mean Squared error is {}".format(rmse))


# In[ ]:




