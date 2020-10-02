#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Introduction

# Firstly we will import our data.

# In[ ]:


data = pd.read_csv("../input/pyramid-scheme-profit-or-loss/pyramid_scheme.csv")
data.head()


# As it is seen in the output we have 4 parameters (cost_price,profit_markup,_depth_of_tree,sales_commission) and one output as called profit.
# * But if we take a look to cost_price and sales_commision, they are constant in each row. So we should ignore them.
# * Otherwise they could mislead our normalization in next steps.
# * We do not need Unnamed: 0 column, since it is basicly ID and it is already founded in np array. So we can delete that column by using drop() function.
# * Drop function's inputs are respectively ;
#     1. ["name of column/row"] ,
#     2. axis = 1 (for dropping column) or 0 (for dropping row) ,
#     3. inplace = True (writes over data) or False (drop it for once)

# In[ ]:


data.drop(["Unnamed: 0","cost_price","sales_commission"] , axis = 1, inplace = True) #axis = 1 for column, axis = 0 for row deleting !!!
data.head()


# In this Kernel, we will do only Logistic Regression for whether job has profit or not.
# * Thus, we need to convert profit column to binary, to detect profit(1) or loss(0).

# In[ ]:


data.profit = [1 if money>0 else 0 for money in data.profit]

sns.countplot(x="profit", data=data) #to visualize #of profits and losses in barchart by usign seaborn lib.
data.loc[:,'profit'].value_counts()


# 130 out of 500 pyramid scheme businesses just succeed to make a profit.
# * Every requirements are prepared for regression. Let's name them.

# In[ ]:


y = data.profit.values
x_data = data.drop(["profit"],axis = 1)


# # Logistic Regression

# Before starting to train and test, we need to normalize our features.
# * Normalizing means fitting features between 0 and 1.
# * Since, some of the features could be misleading. Might be too high and predominate others or exact opposite.
# * It is a basic algebra formula, (x - xmin)/(xmax-xmin).

# In[ ]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head()


# So far so good, now we can split our data to train and test.
# * We will train some portion of features then control whether it is correct or not by comparing with tests.
# * test_size is selected for what percentages of x data would separated as test.
# * random_state = 42 , because function will select same index

# In[ ]:


from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
x_train.head()


# Finally, we can apply Logistic Regression Model.

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T , y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

