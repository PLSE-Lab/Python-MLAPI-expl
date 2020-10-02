#!/usr/bin/env python
# coding: utf-8

# #### **Default Cell For Kaggle Notebook. Load ( Numpy , Pandas , Dataset ( train.csv , test.csv , sample-submission.csv) Automatically**

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


# # Read Dataset

# In[ ]:


print( os.getcwd() ) # Get Current Working Directory


# ***Dataset Path /kaggle/input/support-ml-competition-online-news-popularity/train.csv
# So We should change working directory to***
# 
# /kaggle/input/support-ml-competition-online-news-popularity/

# In[ ]:


os.chdir('/kaggle/input/support-ml-competition-online-news-popularity')

print( os.getcwd() ) # Get Current Working Directory


# # Read Dataset

# In[ ]:


train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# # Split train to ( X , Y ) - ( Input-Features - Target )

# In[ ]:


X , Y = train.drop('abs_title_sentiment_polarity' , axis = 1)  , train['abs_title_sentiment_polarity']
# Drop target column (abs_title_sentiment_polarity) From X , Assign it to Y


# # Linear Regression With Gradient Descent

# ![Linear Regression](https://i.stack.imgur.com/lxwSs.png)

# ## Since Numpy optimize vectors and matrices operations.
# ## Therefore, we should use a vectorized version of the ( cost function ) and (gradient)

# In[ ]:


def compute_cost( m , h , y ): # The function should return cost
    pass 
    # ( m ) denotes to number of rows, (h) our hypothesis of target (y), the true value of target (y)


# In[ ]:


def hypothesis(theta , x): # Function should return hypothesis
    pass


# In[ ]:


def gradient_descent(): # The function should return 1 step of the  gradient descent 
    pass


# In[ ]:


def fit_linear_regression(): # The function should return ( theta , cost -  List of cost at each iteration )
    pass


# In[ ]:


def predict(test_data , theta): # The function should return the predicted value of (abs_title_sentiment_polarity)
    pass


# In[ ]:


def save_to_csv(y_pred): # The function should save the predicted value of (abs_title_sentiment_polarity) to csv file as same as ( sample_submission format)
    pass

