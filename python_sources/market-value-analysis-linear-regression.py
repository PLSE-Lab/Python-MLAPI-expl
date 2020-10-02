#!/usr/bin/env python
# coding: utf-8

# # Refer :
# 
# For conerting data into quantiative - http://pbpython.com/categorical-encoding.html

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


# Path of the file to read
diabetes_file = '../input/MarketValue.csv'
data = pd.read_csv(diabetes_file)
data.head()


# In[ ]:


data.shape


# In[ ]:


data.drop(["name"], axis=1, inplace=True)


# In[ ]:


#Handling unnamed columns in python dataframe
data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
#1st column had no significance as it contains only ids so dropping that column
data.drop(["a"], axis=1, inplace=True)
data.drop(["rank"], axis=1, inplace=True)
data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['profits'].skew()


# In[ ]:



data.profits.isnull().values.any()
data.profits.isnull().sum()
data['profits'] = data['profits'].fillna((data['profits'].median()))


# In[ ]:


data.dtypes
obj_df = data.select_dtypes(include=['object']).copy()
obj_df.head()


# # You can use in pandas is to convert a column to a category, & then use those category values for your label encoding:

# In[ ]:


data["country"] = data["country"].astype('category')
data["category"] = data["category"].astype('category')
data.dtypes


# In[ ]:


data["country"] = data["country"].cat.codes
data["category"] = data["category"].cat.codes
data.head()


# In[ ]:



# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X=data.drop(['marketvalue'], axis=1)
Y=data['marketvalue']

x_train, x_test , y_train , y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
print('Training X Shape:', x_train.shape)
print('Training Y Shape:', y_train.shape)
print('Testing X Shape:', x_test.shape)
print('Testing Y Shape:', y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

linear_reg.fit(x_train,y_train)

y_predict=linear_reg.predict(x_test)


# In[ ]:


linear_reg.score(x_test,y_test)


# In[ ]:


y_predict


# In[ ]:


df1


# In[ ]:





# In[ ]:


a=np.array(y_test)


# In[ ]:


a


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2_score = linear_reg.score(x_test, y_test)

print('Sqrt MSE : ',np.sqrt(mse))
print('R2 Score : ',r2_score)


# In[ ]:


linear_reg.rsquared


# In[ ]:





# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# 

# 

# 

# 

# 

# 
