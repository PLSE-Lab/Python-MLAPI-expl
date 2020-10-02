#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


data.Age.value_counts()


# In[ ]:


data.City_Category.value_counts()


# In[ ]:


data.Stay_In_Current_City_Years.value_counts()


# In[ ]:


data.Marital_Status.value_counts()


# In[ ]:


data.Product_Category_1.value_counts()


# In[ ]:


data.isnull().sum()


# In[ ]:


# filling the null entries with 0

data.fillna(0, inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Encoding the categorical variables

label_encoder_gender = LabelEncoder()
data.iloc[:, 2] = label_encoder_gender.fit_transform(data.iloc[:, 2])


# In[ ]:


data.head()


# In[ ]:


# dummy encoding

df_age = pd.get_dummies(data.Age, columns=['Age'], drop_first=True, prefix='C')


# In[ ]:


df_age.head()


# In[ ]:


df_city = pd.get_dummies(data.City_Category, columns=['City_Category'], drop_first=True, prefix='C')


# In[ ]:


df_city.head()


# In[ ]:


df_stay = pd.get_dummies(data.Stay_In_Current_City_Years, columns=['Stay_In_Current_City_Years'], drop_first=True, prefix='C')


# In[ ]:


df_stay.head()


# In[ ]:


# Combining the original data set to with the dummy datasets

data_final = pd.concat([data, df_age, df_city, df_stay], axis=1)
data_final.head()


# In[ ]:


# Dropping the the original categorical columns 

data_final.drop(['User_ID', 'Product_ID', 'Age', 'City_Category', 'Stay_In_Current_City_Years'], axis=1, inplace=True)


# In[ ]:


data_final.head()


# In[ ]:


X = data_final.drop('Purchase', axis=1)
y = data_final.Purchase


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


# Building regression model on the data to predict 'Purchase' 

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


print(r2_score(y_test, y_pred))


# In[ ]:




