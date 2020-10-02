#!/usr/bin/env python
# coding: utf-8

# ### Preliminary setup

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Data import

# In[ ]:


data_path = '../input/Admission_Predict.csv'
df = pd.read_csv(data_path)
df.head()


# In[ ]:


print(len(df.columns),"columns are:")
for x in df.columns:
    print(x+ ',')


# Better to change the name of the last column - it's gonna be used.

# In[ ]:


df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# In[ ]:


print(df.info ())


# 400 observations, no missing values.

# ### Brief Visualisation

# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(10, 10))
sb.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# So, most important features for admission: CGPA, GRE, TOEFL.

# ### Small model: train and test samples

# In[ ]:


y = df['Chance of Admit']
features = ['GRE Score', 'TOEFL Score', 'CGPA']
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3, random_state=1)


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X, train_y)
lr_predicts = lr.predict(val_X)
from sklearn.metrics import mean_squared_error
lr_mse = mean_squared_error(lr_predicts, val_y)
print(lr_mse)


# From the linear regression: MSE=0.004301 .
# Let's try logistic regression instead.

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
train_y_01 = [1 if each > 0.5 else 0 for each in train_y]
logr = LogisticRegression()
logr.fit(train_X, train_y_01)
logr_predicts = logr.predict(val_X)
logr_mse = mean_squared_error(logr_predicts, val_y)
print(logr_mse)


# From the logistic regression: MSE = 0.1024575 (large than with the simple linear regression).
# Obviously, we lost a lot of info when recoded y into a binary variable.
# 
# Next, let's try DecisionTreeRegressor.
# 
# ### DecisionTreeRegressor

# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)
tree.fit(train_X, train_y)
tree_predicts = tree.predict(val_X)
tree_mse = mean_squared_error(val_y, tree_predicts)
print(tree_mse)


# DecisionTreeRegressor is better than LogisticRegression, but worse than the LinearRegression.

# In[ ]:


test_data_path = '../input/Admission_Predict_Ver1.1.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = lr.predict(test_X)
output = pd.DataFrame({'Serial No.': test_data['Serial No.'],
                       'Chance of Admit': test_preds})
output.to_csv('submission.csv', index=False)

