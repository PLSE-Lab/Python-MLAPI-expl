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


# 1. Let's open and load the train.csv

# In[ ]:


train= pd.read_csv("../input/first-challenge-find-the-output-number/train.csv")
test = pd.read_csv("../input/first-challenge-find-the-output-number/test.csv")


# Observe  a few lines

# 

# In[ ]:


train.head()


# In[ ]:


test.tail()


# Check data

# In[ ]:


train.info()


# 

# Since the target values float, that is, continues values (not discreet values) this pronlem is a regression problem.
# Let's use some regression methods available in  statsmodels library
# See 
# [1](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html/) https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html
# 

# First, seperate input and target columns

# In[ ]:


## X usually means our input variables (or independent variables)
X = train.iloc[:,:-2]
## Y usually means our output/dependent variable
y_A= train.iloc[:,6:7]
y_B= train.iloc[:,7:]
print("X", X.head())
print("y_A",y_A.head())
print("y_B",y_B.head())


# In[ ]:





# In[ ]:





# In[ ]:


import statsmodels.api as sm
#X = sm.add_constant(X)
model_A = sm.OLS(y_A, X).fit()
print("modelA",model_A.summary())

model_B = sm.OLS(y_B, X).fit()
print("modelB",model_B.summary())


# In[ ]:


X= test.iloc[:,1:]
predictionsForA = model_A.predict(X)
predictionsForB = model_B.predict(X)


# In[ ]:





# In[ ]:


predictionsForA.head()


# In[ ]:


my_submission = pd.DataFrame({ 'ID': test.ID,'A': predictionsForA, 'B': predictionsForB})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()


# Everything runs smoothly, but the problem is you can't see your file anywhere in this page, nor in your Profile, Kernels tab, nowhere! This is because you haven't commited your notebook yet. To do that, click the Commit button - as I write it, this is a light-blue button in the top-right corner of my notebook page, in the main pane. (There is also a right pane with Sessions, Versions etc. You can ignore it for now). It may take a minute for the Kaggle server to publish your notebook. 
# 
# When this operation is done, you can go back by clicking '<<' button in the top-left corner. Then you should see your notebook with a top bar that has a few tabs: Notebook, Code, Data, Output, Comments, Log ... Edit Notebook. Click the Output tab. You should see your output csv file there, ready to download!

# ![](http://)

# In[ ]:


from IPython.display import Image
Image("../input/submission/submission.png")


# ![1](../submission/submission.png)
