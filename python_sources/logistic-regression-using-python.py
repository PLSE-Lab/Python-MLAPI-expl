#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import Essential Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression #used to import linear & logistic reg


# # > **Reading a file**

# In[ ]:


df=pd.read_csv("../input/titanic/train_and_test2.csv")
df=df.dropna()# removes null values(if any)
df.shape #gives no.of rows & coloumns
df


# Here,there are 1307 rows and 28 coloumns of non null values.

# In[ ]:


df.keys() #it gives name of the coloumns 


# Let us consider,'Passengerid', 'Age', 'Fare', 'Sex', 'sibsp' as independent data from the dataset

# And '2urvived' as dependent data from the dataset

# In[ ]:


x=df[['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp']].values #independent variable
y=df[['2urvived']].values #dependent variable


# # Training the model with LinearRegression

# In[ ]:


lr_model=LinearRegression()
lr_model.fit(x,y)


# In[ ]:


y_pred=lr_model.predict(x)


# # Sigmoid Function

# Logistic Reg deals with sigmoid function.Which in turns formulates to be=1/1+e^(-output)

# In[ ]:


exp=np.exp(-y_pred)+1
log=1/exp #1/1+e^(-y_pred)


# # Training with Logistic Regression

# The given data set is in the form of numbers.
# 

# It has to be converted either into True or False in logistic reg.

# In[ ]:


y_con=df['2urvived']>0 #for all the values >0 it shows true.
y_con


# All 0's are converted to false & all 1's are converted to true

# For total no.of true & false in the data,we use the following code.

# In[ ]:


df["tf"]=df['2urvived']>0
df.tf.value_counts() #gives no of true & false.


# # Importing Logistic Regression

# In[ ]:


log_reg=LogisticRegression()
log_reg.fit(x,y_con) # machine is trained with logistic regression


# In[ ]:


log_reg.predict([[2,38.0,71.2833,1,1]]) #predicting the values for independent values


# To find the probability for above prediction,

# In[ ]:


log_reg.predict_proba([[2,38.0,71.2833,1,1]])


# **Hence,we have performed logistic reg on predicting th probability of survived data based on independent values given.**

# In[ ]:




