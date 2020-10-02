#!/usr/bin/env python
# coding: utf-8

# # If you like the work in notebook please upvote

# # This is real world problem where we are to build a model which can predict how much loss we might face from a particular customer ( target feature 'loss' )
# 

# # Each Record in the dataset describes different attributes of a customer so it has both categorical and numeric features

# In[ ]:


# this section is just to import the dataset file , so don't worry about it

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# let's just read our data
df=pd.read_csv('/kaggle/input/regression-116-categorical-16-numeric-features/insur.csv',index_col=0)
df.head()


# In[ ]:


# let's see how many rows and columns we have in our dataset

df.shape

# we have 117794 rows and 131 columns


# In[ ]:


# let's look at the data in depth
df.describe()


# In[ ]:


# let's see if we have any missing values in our dataset

df.isnull().sum()
df.isna().sum()
# As we can see there are no missing values 


# In[ ]:


# let's create our x and y 
y=df.iloc[:,-1]
x=df.iloc[:,:-1]
x


# In[ ]:


# let's split our data into train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=2)


# In[ ]:


# let's define our model , we will use CatBoostRegressor because it can handle categorical variable so we need not
# preprocess our data 

from catboost import CatBoostRegressor

model=CatBoostRegressor()


# In[ ]:


# let's seperate our categorical columns
cat_col=x_train.select_dtypes(include='object').columns
cat_col


# In[ ]:


# let's fit our model to the training dataset

model.fit(x_train,y_train,cat_features=cat_col)


# In[ ]:


# let's check training accuracy of our model

model.score(x_train,y_train)
# as we can see our model is still underfitting and has 63%  training accuracy


# In[ ]:


# let's check our test accuracy

model.score(x_test,y_test)

# model has 58% test accuracy which is much like tossing a coin


# # Check my other notebook for the optimized model, if you like this notebook please upvote
