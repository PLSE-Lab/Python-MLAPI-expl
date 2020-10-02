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


df=pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df[df.duplicated(keep=False)]


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.iloc[12320:12325,:]


# In[ ]:


df=df.drop_duplicates()


# In[ ]:


df.iloc[12320:12325,:]


# In[ ]:


#Since there are so many categorical column analyze what are the coluns are required.


# In[ ]:


df.columns


# In[ ]:


df['Indicator Category'].value_counts()


# In[ ]:


df.Indicator.nunique()


# In[ ]:


df.Indicator.value_counts()


# In[ ]:


#Assuming/understanding that based on indicator it is already categorized and given as indocator category. So drop indaicator column


# In[ ]:


df=df.drop(columns='Indicator')


# In[ ]:


df.Year.value_counts()


# few years are given in rage. either we can group it as any of the existing year like 2008 - 2012 as 2012 
# here i am ignoring all the rows

# In[ ]:


#considering the rows which is not having the -(hyphen)
df = df[~df.Year.str.contains("-")]


# In[ ]:


df.Year.value_counts()


# In[ ]:


df.Gender.value_counts()


# In[ ]:


df['Race/ Ethnicity'].value_counts()


# In[ ]:


df.Place.value_counts()


# In[ ]:


df['BCHC Requested Methodology'].value_counts()


# In[ ]:


df=df.drop(columns='BCHC Requested Methodology')


# In[ ]:


df.columns


# In[ ]:


df.Source.value_counts()


# In[ ]:


df.Notes.value_counts()


# In[ ]:


df.Methods.value_counts()


# In[ ]:


#unable find any relevance - so removing the columns
df= df.drop(columns=["Source","Notes","Methods"])


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.Place


# In[ ]:


df["city_Info"] = df.Place.apply(lambda x : x[-2:])


# In[ ]:


df["city_Info1"] = df.Place.apply(lambda x : x[:-4])


# In[ ]:


df.city_Info1.head()


# In[ ]:


df["city_Info1"].value_counts()


# In[ ]:


df["city_Info"].head()


# In[ ]:


df["city_Info"].value_counts()


# In[ ]:


df.head()


# In[ ]:


df=df.drop(columns='Place')


# In[ ]:


df.head()


# In[ ]:


df.Value.isna().sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


#encoding 
df_column_cat = df.select_dtypes(exclude=np.number).columns


# In[ ]:


encoded_cat_col = pd.get_dummies(df[df_column_cat])


# In[ ]:


df_final = pd.concat([df['Value'],encoded_cat_col], axis = 1)


# In[ ]:


X = df_final.drop(columns=['Value'])
y = df_final['Value']


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
model = LinearRegression()
model.fit(X_train,y_train)


# In[ ]:


print("*****coefficient valuessss",model.coef_)
print("*****intercept iss",model.intercept_)


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


train_predict = model.predict(X_train)

mae_train = mean_absolute_error(y_train,train_predict)

mse_train = mean_squared_error(y_train,train_predict)

rmse_train = np.sqrt(mse_train)

r2_train = r2_score(y_train,train_predict)

mape_train = mean_absolute_percentage_error(y_train,train_predict)


# In[ ]:


test_predict = model.predict(X_test)

mae_test = mean_absolute_error(test_predict,y_test)

mse_test = mean_squared_error(test_predict,y_test)

rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))

r2_test = r2_score(y_test,test_predict)

mape_test = mean_absolute_percentage_error(y_test,test_predict)


# In[ ]:


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TRAIN: Mean Absolute Error(MAE): ',mae_train)
print('TRAIN: Mean Squared Error(MSE):',mse_train)
print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)
print('TRAIN: R square value:',r2_train)
print('TRAIN: Mean Absolute Percentage Error: ',mape_train)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TEST: Mean Absolute Error(MAE): ',mae_test)
print('TEST: Mean Squared Error(MSE):',mse_test)
print('TEST: Root Mean Squared Error(RMSE):',rmse_test)
print('TEST: R square value:',r2_test)
print('TEST: Mean Absolute Percentage Error: ',mape_test)


# In[ ]:


sns.scatterplot(y_train,train_predict)


# There is no linearity between actual data and predicted data.NOT fit for linear regresson model.
