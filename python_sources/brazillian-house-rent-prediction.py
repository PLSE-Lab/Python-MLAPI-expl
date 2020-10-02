#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



# I choose version 2 from the dataset
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head(2)


# In[ ]:


#I turned the categorial data to binary .which is 0 and 1
data['animal']=data['animal'].replace(to_replace ="acept", 
                 value =1) 
data['animal']=data['animal'].replace(to_replace ="not acept", 
                 value =0)

data['furniture']=data['furniture'].replace(to_replace ="furnished", 
                 value =1) 
data['furniture']=data['furniture'].replace(to_replace ="not furnished", 
                 value =0)


# In[ ]:


data.tail(3)


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.rename(columns={"rent amount (R$)": "rent"},inplace=True)


# In[ ]:


print("Find most important features relative to target")
corr = data.corr()
corr.sort_values(["rent"], ascending = False, inplace = True)
print(corr.rent)


# From the above output we can see that
# 
# 
# fire insurance (R$)  =  0.987343
# 
# 
# bathroom              = 0.668504
# 
# 
# parking spaces       =  0.578361
# 
# 
# rooms                =  0.541758
# 
# 
# total (R$)          =   0.264490
# 
# 
# area                =   0.180742
# 
# 
# furniture            =  0.164235
# 
# 
# property tax (R$)   =   0.107884
# 
# 
# animal              =   0.067754
# 
# 
# hoa (R$)            =   0.036490
# 
# 
# 

# In[ ]:


data['floor']=data['floor'].replace(to_replace ="-", 
                 value =0)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data[['area','rooms','bathroom','parking spaces','floor','animal','furniture','property tax (R$)','fire insurance (R$)']],data.rent,test_size=0.1)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x_test, y_test);


# In[ ]:


rf.predict(x_test)


# In[ ]:


rf.score(x_test,y_test)


# I got the score upto 99 percent which means our model is accurate . pretty much !!!

# In[ ]:




