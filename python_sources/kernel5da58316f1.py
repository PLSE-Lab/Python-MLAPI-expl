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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")


# In[ ]:


df.head()


# In[ ]:


df['Notes'].value_counts()


# In[ ]:


df['Year']


# In[ ]:


df.columns


# In[ ]:


cols_to_Encode = ['Gender','Race/ Ethnicity','Indicator Category']
continuous_cols = ['Value']


# In[ ]:


encoded_cols = pd.get_dummies(df[cols_to_Encode])


# In[ ]:


df_final = pd.concat([encoded_cols,df[continuous_cols]], axis = 1)


# In[ ]:


y = df_final['Value']
x = df_final.drop(columns = 'Value')


# In[ ]:


df_final.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


model = LinearRegression()


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(x,y,test_size = 0.3)
df_final.columns.dtype


# In[ ]:


df_column_category = df_final.select_dtypes(exclude=np.number).columns
df_column_category


# In[ ]:


df_final['Year'].value_counts()


# In[ ]:


df_final.isna().sum()


# In[ ]:


df_final['Value'].value_counts()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_final.Value.plot(kind="box")


# In[ ]:


df_final.Value.fillna(df.Value.median(),inplace = True)


# In[ ]:


model.fit(train_X,train_Y)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


train_predict = model.predict(train_X)


# In[ ]:


test_predict = model.predict(test_X)


# In[ ]:


##MAE
print(mean_absolute_error(train_Y,train_predict))
##MAE
print(mean_absolute_error(test_Y,test_predict))


# In[ ]:


##MSE
print(mean_squared_error(train_Y,train_predict))
##MSE
print(mean_squared_error(test_Y,test_predict))

