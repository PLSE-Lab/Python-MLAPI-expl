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


dataf=pd.read_csv("../input/bike_share.csv")


# In[ ]:


dataf.head()


# In[ ]:


dataf.info()


# In[ ]:


dataf.describe().T


# In[ ]:


dataf.duplicated().sum()


# In[ ]:


dataf.shape


# In[ ]:


dataf.drop_duplicates(inplace=True)


# In[ ]:


dataf.duplicated().sum()


# In[ ]:


dataf.isna().sum()


# In[ ]:


dataf["temp"].unique()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
dataf.season.plot(kind="box")


# In[ ]:


dataf.windspeed.plot(kind="box")





# In[ ]:


dataf.casual.plot(kind="box")


# In[ ]:


dataf.registered.plot(kind="box")


# In[ ]:


dataf["count"].plot(kind="box")


# In[ ]:


dataf["count"].value_counts()


# In[ ]:


dataf.corr()


# In[ ]:


train_X = dataf.drop(columns=["temp","casual","registered"])


# In[ ]:


train1_Y = dataf["casual"]
train2_Y = dataf["registered"]
train3_Y = dataf["count"]


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model


# In[ ]:


a = cross_val_score(model, train_X, train_Y, cv=5, scoring='neg_mean_squared_error')


# In[ ]:



np.mean(np.sqrt(np.abs(a)))


# In[ ]:


model.fit(train_X, train1_Y)
        
    #Predict training set:
dtrain_predictions = model.predict(train_X)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(train_X, train1_Y, test_size=0.3, random_state=123)


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


train_predict = model.predict(X_train)


# In[ ]:


test_predict = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# In[ ]:





# In[ ]:


print("Train:",mean_absolute_error(Y_train,train_predict))
print("Test:",mean_absolute_error(Y_test,test_predict))


# In[ ]:


print('r2 train',r2_score(Y_train,train_predict))
print('r2 test',r2_score(Y_test,test_predict))


# In[ ]:


import seaborn as sns


# In[ ]:


for i in dataf.columns:
        sns.pairplot(data=dataf,x_vars=i,y_vars='count')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




