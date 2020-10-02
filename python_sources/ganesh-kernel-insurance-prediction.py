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


dataf=pd.read_csv("../input/insurance.csv")
datad=pd.read_csv("../input/insurance.csv")


# In[ ]:


dataf.head()


# In[ ]:


dataf.shape


# In[ ]:


gender = {'male': 1,'female': 2} 


# In[ ]:


dataf.sex = [gender[item] for item in dataf.sex] 


# In[ ]:


dataf.head()


# In[ ]:


smoke = {'yes': 1,'no': 2} 


# In[ ]:


dataf.smoker = [smoke[item] for item in dataf.smoker] 


# In[ ]:


dataf.region.unique()


# In[ ]:


reg = {'southwest' :1, 'southeast' : 2, 'northwest' : 3, 'northeast' : 4}


# In[ ]:


dataf.region = [reg[item] for item in dataf.region] 


# In[ ]:


dataf.head()


# In[ ]:


dataf.corr()


# In[ ]:


for i in dataf.columns:
  print('{} is unique: {}'.format(i, dataf[i].is_unique))


# In[ ]:


dataf.index.values


# In[ ]:


dataf.isna().sum()


# In[ ]:


dataf


# In[ ]:


datad.head()


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X= dataf.drop(["expenses","region","children","sex","bmi"],axis=1)
Y = dataf["expenses"]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


Y_train.shape


# In[ ]:


Y_test.shape


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# In[ ]:


train_predict = model.predict(X_train)


# In[ ]:


test_predict = model.predict(X_test)


# In[ ]:


print("Train:",mean_absolute_error(Y_train,train_predict))


# In[ ]:


print("Test:",mean_absolute_error(Y_test,test_predict))


# In[ ]:


print("Train:",mean_squared_error(Y_train,train_predict))


# In[ ]:


print("Test:",mean_squared_error(Y_test,test_predict))


# In[ ]:


print('RMSE train',np.sqrt(mean_squared_error(Y_train,train_predict)))
print('RMSE test',np.sqrt(mean_squared_error(Y_test,test_predict)))


# In[ ]:


print('r2 train',r2_score(Y_train,train_predict))
print('r2 test',r2_score(Y_test,test_predict))


# In[ ]:


import seaborn as sns


# In[ ]:


for i in dataf.columns:
        sns.pairplot(data=dataf,x_vars=i,y_vars='expenses')


# In[ ]:


X.head()

