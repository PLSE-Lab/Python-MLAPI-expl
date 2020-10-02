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


# **Import challenge data**
# 
# We import the data using pandas

# In[ ]:


bike = pd.read_csv("../input/train.csv")
bike.head()


# In[ ]:


bike.describe()


# Now we will check our outcome (cnt) to see if it is a normally distributed variable

# In[ ]:


import seaborn as sns

sns.distplot(bike['cnt'])


# As we can see, our outcome does not follow a normal distribution. It has two picks,

# In[ ]:


g = sns.FacetGrid(bike, row="workingday", col="holiday", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')


# In[ ]:


g = sns.FacetGrid(bike, row="workingday", col="season", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')


# In[ ]:


g = sns.FacetGrid(bike, row="weathersit", col="season", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')


# We will divide the train dataset into two sets, one for the independent variables (X) and one for the outcome (y).

# In[ ]:


X_train = bike.drop(columns="cnt")
y_train = bike['cnt']


# Now we will proceed to run a linear regression to show how models have to be ran in this challenge. 
#  
# **Please be aware that this model is not the ideal one because the outcome is not normally distributed !!!**

# In[ ]:


from sklearn.linear_model import LogisticRegression

# Initialize the predictive model object
mod_logistic = LogisticRegression()

# Train the model using the training sets
mod_logistic.fit(X_train, y_train)


# In[ ]:


# Make predictions using the testing set
pred = mod_logistic.predict(X_train)

sns.distplot(pred)


# In[ ]:


sns.scatterplot(pred, y_train)


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


test['cnt'] = mod_logistic.predict(test)
test.head()


# In[ ]:


res = test[(['id','cnt'])]
res.head()


# In[ ]:


### run this to generate the prediction file. Change each time the name by adding info related to the model and the version,
### you must upload this files into the kaggle platform (on our competition page) so this prediction can enter the challenge.

# res.to_csv("prediction_lm_v1.0.csv")

