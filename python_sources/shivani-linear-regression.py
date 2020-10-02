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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


insurance_data = pd.read_csv('../input/insurance.csv')


# In[ ]:


insurance_data.info()


# In[ ]:


insurance_data.head()


# In[ ]:


insurance_data["sex"].value_counts()


# In[ ]:


insurance_data["sex"] = insurance_data.sex.replace({'male':1,'female':2})


# In[ ]:


insurance_data["smoker"] = insurance_data.smoker.replace({'yes':1,'no':0})


# In[ ]:


insurance_data["region"].value_counts()


# In[ ]:


insurance_data["region"] = insurance_data.region.replace({'southeast':1,'southwest':2,'northwest':3,'northeast':4})


# In[ ]:


insurance_data.tail()


# In[ ]:


sns.pairplot(data=insurance_data)


# In[ ]:


insurance_data.corr()


# In[ ]:


From the above plot and correlation outputs, its known that age and region variables are negatively correlated with expenses.
So, leaving these variables and making the model.


# In[ ]:


insurance_data.info()


# In[ ]:


Y = insurance_data["expenses"]
x = insurance_data.drop(columns = ["expenses","sex","region"])


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(x,Y,test_size = 0.3,random_state = 42)


# In[ ]:


train_x.shape


# In[ ]:


test_x.shape


# In[ ]:


model = LinearRegression()
model.fit(train_x,train_y)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


train_x.columns


# Based on the output, the intercept value is -12472.897. And the coefficients B1, B2,B3 and B4 are 261.835, 333.484, 433.500 and 23626.098 respectively. The regression model can be written, y = -12472.897 + 261.835(age) + 333.484(bmi) + 433.500(children) + 23626.098(smoker).

# In[ ]:


train_predict = model.predict(train_x)


# In[ ]:


test_predict = model.predict(test_x)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE - Train_data :" ,mean_squared_error(train_y,train_predict))
print("MSE - Test_tata :" ,mean_squared_error(test_y,test_predict))
print("MAE - Train_data :" ,mean_absolute_error(train_y,train_predict))
print("MAE - Test_data :" ,mean_absolute_error(test_y,test_predict))
print("R2 - Train_data :" ,r2_score(train_y,train_predict))
print("R2 - Test_data :" ,r2_score(test_y,test_predict))
print("Mape - Train_data:" , np.mean(np.abs((train_y,train_predict))))
print("Mape - Test_data:" ,np.mean(np.abs((test_y,test_predict))))


# In[ ]:




