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
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.shape


# In[ ]:


df.rename(columns={"Indicator Category":"Indicator_Category"},inplace=True)
df.Indicator_Category.unique()


# In[ ]:


df=df.drop(["Source","Methods","Notes","BCHC Requested Methodology","Indicator"],axis=1)
df.head(30)


# In[ ]:


df.Indicator_Category.unique()


# In[ ]:


df.rename(columns={"Race/ Ethnicity":"Race"},inplace=True)
df.Race.unique()


# In[ ]:


df.Place.unique()


# In[ ]:


df["Place_Info"] = df.Place.apply(lambda x : x[-2:])


# In[ ]:


df["Place_Info"].value_counts()


# In[ ]:


df=df.drop(["Place"],axis=1)
df.head(30)


# In[ ]:


df.Value.plot(kind="box")


# In[ ]:


df.Value.value_counts()


# In[ ]:


df.loc[df.Value==0.000000,"Value"]=np.NAN


# In[ ]:


df["Value"] = df.groupby("Place_Info").Value.transform(lambda x : x.fillna(x.median()))


# In[ ]:


df.Value.isna().sum()


# In[ ]:


df.head()


# In[ ]:


df_column_numeric = df.select_dtypes(include=np.number).columns


# In[ ]:


df_column_category = df.select_dtypes(exclude=np.number).columns


# In[ ]:


df_category_onehot = pd.get_dummies(df[df_column_category])


# In[ ]:


df_final = pd.concat([df_category_onehot,df[df_column_numeric]], axis = 1)


# In[ ]:


df_final.head()


# In[ ]:


x= df_final.drop(["Indicator_Category_Cancer"],axis=1)
x.head()


# In[ ]:


y=df_final["Indicator_Category_Cancer"]
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
model = LinearRegression()
model.fit(x_train,y_train)  # Providing the training values to find model's intercept and slope


# In[ ]:


model.intercept_


# In[ ]:


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# In[ ]:


train_aberror=metrics.mean_absolute_error(y_train,y_train_pred)
test_aberror=metrics.mean_absolute_error(y_test,y_test_pred)

print("mean_absolute_error train",train_aberror)
print("mean_absolute_error test",test_aberror)

train_sqerror=metrics.mean_squared_error(y_train,y_train_pred)
test_sqerror=metrics.mean_squared_error(y_test,y_test_pred)

print("mean_squared_error train",train_sqerror)
print("mean_squared_error test",test_sqerror)

train_rootsqerror=np.sqrt(train_sqerror)
test_rootsqerror=np.sqrt(test_sqerror)

print("squared root of mean_squared_error train",train_rootsqerror)
print("squared root of mean_squared_error test",test_rootsqerror)


# In[ ]:




