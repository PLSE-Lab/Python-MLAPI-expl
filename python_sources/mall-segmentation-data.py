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
import  seaborn as sns
df=pd.read_csv('../input/Mall_Customers.csv')
df


# In[ ]:


df.count()


# In[ ]:



df.columns=['CustomerID','Gender','Age','Annualincomeindollar','spendingscore']
df


# In[ ]:


y=df.Annualincomeindollar


# In[ ]:


x=df[["Age","spendingscore"]]


# In[ ]:


x.describe()


# In[ ]:


y.describe()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
df=DecisionTreeRegressor(random_state=1)
df.fit(x,y)


# In[ ]:


print("make predicion for the following five house ")
print(x.head())


# In[ ]:


print("the prediction are ")
print(df.predict(x.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predict_annualincomeindollar=df.predict(x)
mean_absolute_error(y,predict_annualincomeindollar)


# the above we computed called as "IN SAMPLE" score

# # THEORY OF VALIDATION
# The Problem with "In-Sample" Scores
# The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it. Here's why this is bad.
# 
# ## EXAMP LE OF A DATA SET
# Imagine that, in the large real estate market, door color is unrelated to home price.
# 
# However, in the sample of data you used to build the model, all homes with green doors were very expensive. The model's job is to find patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with green doors.
# 
# Since this pattern was derived from the training data, the model will appear accurate in the training data.
# 
# But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.
# 
# Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. 
# ### This data is called validation data.
# 
# 

# The scikit-learn library has a function train_test_split to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate mean_absolute_error.
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)


# In[ ]:


df=DecisionTreeRegressor()
df.fit(train_x,train_y)
val_prediction=df.predict(val_x)
print (mean_absolute_error(val_y,val_prediction))


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
def get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1)
    model.fit(train_x,train_y)


# th above half code is of decision tree regression 
# we have forest tree regression to improve the overfitting and underfitting 
# we will see after some time

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_x,train_y)
superpredict=forest_model.predict(val_x)
print(mean_absolute_error(val_y,superpredict))


# as you see it goes t 23 to 17 when we predit from decision tree to 

# In[ ]:




