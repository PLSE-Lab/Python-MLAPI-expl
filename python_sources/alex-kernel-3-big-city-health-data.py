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


df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.describe().T


# In[ ]:


df["Indicator Category"].value_counts()


# In[ ]:


ind = df["Indicator"]
val = []
n= 0
for i in ind:
    if 'hiv' in i or 'HIV' in i or 'aids' in i or 'AIDS' in i:
        val.append(i)
        n+=1
n


# In[ ]:


df["Year"].value_counts()


# In[ ]:


df['Gender'].value_counts()


# In[ ]:


df['Race/ Ethnicity'].value_counts()


# In[ ]:


df['Place'].value_counts()


# In[ ]:


df['BCHC Requested Methodology'].value_counts()


# In[ ]:


df['Source'].value_counts()


# In[ ]:


df['Methods'].value_counts()


# In[ ]:


df['Notes'].value_counts()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.shape


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


df['Value'].mean()


# **Replacing Null values with mean of value column**

# In[ ]:


df['Value'].fillna(df['Value'].mean(),inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.columns


# **Removing 5 columns as I couldn't find out any patterns out of it.**. TBD

# In[ ]:


df.drop(columns=['Indicator','BCHC Requested Methodology', 'Source', 'Methods', 'Notes'],inplace=True)


# In[ ]:


df.columns


# In[ ]:


df["Year"].value_counts()


# **In values column most of the data falls into years 2010...2014  some values are represented with range instead of single year, as these values are very few in count, I'm just taking the last year in that range so that the categories are less and meaningful now **

# In[ ]:


for val in df['Year']:
    if '-' in val:
        temp = val[5:len(val)]
        df['Year'].replace(val, temp, inplace=True)


# In[ ]:


df["Year"].value_counts()


# **Going to do one-hot encoding as there are categorical colummns**

# In[ ]:


df.columns


# In[ ]:


cat_cols =  df.select_dtypes(exclude=np.number)


# In[ ]:


num_cols = df.select_dtypes(include=np.number)


# In[ ]:


encoded_cat_cols = pd.get_dummies(cat_cols)


# In[ ]:


preprocessed_df = pd.concat([encoded_cat_cols, num_cols], axis=1)


# In[ ]:


preprocessed_df.head()


# **Preprocessing part is done. Now going to build linear regression model with train and test data**

# In[ ]:


x = preprocessed_df.drop(columns='Value')


# In[ ]:


y = preprocessed_df[['Value']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=12)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


train_predict = model.predict(train_x)


# In[ ]:


test_predict = model.predict(test_x)


# **Now the model is ready and it's predicted the target values. Now let's evaluate  the results using Error Metrics**

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


MAE_train = mean_absolute_error(train_y,train_predict)
MAE_test = mean_absolute_error(test_y,test_predict)

MSE_train = mean_squared_error(train_y,train_predict)
MSE_test = mean_squared_error(test_y,test_predict)

RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)

R2_train = r2_score(train_y, train_predict)
R2_test = r2_score(test_y, test_predict)


# In[ ]:


print("MAE of Trained data : ",MAE_train)
print("MAE of Test data : ", MAE_test)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("MSE of Trained Data", MSE_train)
print("MSE of Test Data", MSE_test)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("RMSE of Trained Data", RMSE_train)
print("RMSE of Test Data", RMSE_test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("R2 of train: ", R2_train)
print("R2 of test: ", R2_test)


# In[ ]:





# In[ ]:




