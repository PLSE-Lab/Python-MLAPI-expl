#!/usr/bin/env python
# coding: utf-8

# # Introduction

# 

# 

# ## Importing libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


# ## Loading data

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/input/learn-together/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Read the data

# In[ ]:


#Read the files train.csv and test.csv into variables df_train_data, and df_test_data.
train_data = pd.read_csv('../input/learn-together/train.csv')
test_data = pd.read_csv('../input/learn-together/test.csv')


# ## Exploratary data analysis
# 
# Goals of Excecution Data Analysis
# Size of data
# Prooperties of the target variable (check for issues like high class imbalance, skewed distribution in a regression
# Properties of the features: Finding sime peculiarities and dependecies between features and target variable is always useful
# Generate ideas for feature engineering and future ypothesis
# 

# ### Size of the data

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# 

# 

# In[ ]:


print("train dataset shape "+ str(train_data.shape))
print("test dataset shape "+ str(test_data.shape))


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


#train_data.describe()


# In[ ]:


#test_data.describe()


# In[ ]:


X = train_data.copy()
X = X.drop(columns=['Cover_Type'])
y = train_data[['Cover_Type']]


# In[ ]:


X.columns


# In[ ]:


colorado_features = ['Id', 'Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon']
X = X[colorado_features]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
colorado_model = DecisionTreeRegressor(random_state=1)

# Fit model
colorado_model.fit(X, y)


# In[ ]:


print("Making 5 predictions")
print(X.head())
print("The predictions are")
print(colorado_model.predict(X.head()))


# 

# In[ ]:


print("OK")
print(X,"The predictions are    ", colorado_model.predict(X))
print()


# 

# # Competition Mettric: Categorization Accuracy, percentage of correct predictions

# In[ ]:


def evaluate_metric_score(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]:
        raise Exception("Sizes do not match")
        return 0
    else:
        size = y_true.shape[0]
        matches = 0
        y_true_array = np.array(list(y_true))
        y_pred_array = np.array(list(y_pred))
        for i in range(0, size):
            if y_true_array[i]==y_pred_array[i]:
                matches = matches + 1
        return mathces/size


# 

# In[ ]:


output = pd.DataFrame({'Id': X.index,
                       'Cover_Type': test_preds})
output.to_csv('submission1.csv', index=False)
output.head()

