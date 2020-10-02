#!/usr/bin/env python
# coding: utf-8

# # Introduction to machine learning with scikit-learn
# In this kernel, I will demonstrate how to use existing libraries to build a model for a regression task. You don't need to copy paste the code, just fork this notebook and run in the browser. The kernel is organized as follows. First, I do some very simple exploratory data analysis to understand the data better. Next, I will show some preprocessing steps and model training. Finally, I give some ideas on how to improve the model performance.

# ## Exploratory data analysis
# Exploratory data analysis means visualizing the data from different perspectives to see some patterns and make assumptions about the data. It is often said to be the most important step in Data Science process. I'll keep it very simple visualizing only histograms, but there are many possibilities what to do.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split # utils
from sklearn.metrics import mean_absolute_error # eval metric

# data processing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNet # machine learning

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# After importing the packages, let's read the dataset (using pandas) and show the first 5 rows.

# In[ ]:


# using pandas to read CSV files
df = pd.read_csv('../input/X_train.csv')
y = pd.read_csv('../input/y_train.csv')['PRP']
df_test = pd.read_csv('../input/X_test.csv')
df.head()


# Looks like all variables have discrete and positive values. Let's do some simple univariate visualization.

# In[ ]:


fig, ax = plt.subplots(3,2,figsize=(10,8))
for i,c in enumerate(df):
    sns.distplot(df[c], ax=ax[i // 2][i % 2], kde=False)
fig.tight_layout()


# None of the features are normaly distributed. Perhaps, we should transform them by some nonlinear function such as $log$ or $x^{-1}$.

# In[ ]:


sns.distplot(y)
print(y.head())


# The distribution of the target variable. All values are non-negative  as well.

# ## Data preprocessing and modeling
# So far, we have a very rough idea about the dataset. Before we run any machine learning algorithm, we should preprocess our data, otherwise, the model performance can significantly suffer. The scikit-learn API offers many preprocessing and machine learning tools. Most of the models have two main methods. The first one is **fit** which learns from data and the second is **predict** or **transform** which applies the model on new datasets (without learning). Here we use the class **PolynomialFeatures** to create new variables and **StandardScaler** to normalize the data. Moreover, we use a class **Pipeline** to concisely combine these two operations.

# In[ ]:


# split the data
X, y = df.values, y.values
X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=2018)

# data preprocessing using sklearn Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)), # multiply features together
    ('scale', StandardScaler()), # scale data
])

# fit and apply transform
X_train = pipeline.fit_transform(X_train)
# transform the validation set
X_val = pipeline.transform(X_val)
print('train shape:', X_train.shape, 'validation shape:', X_val.shape)


# After expanding our dataset into 22 features, we can now train our model and evaluate it on the validation set. It is a good practice to compare the results with some very poor prediction such as the average of the target variable, so we know how good we are. We use a linear model with a fancy name *Elastic Net*.  This model is like a linear regression but it adds a penalty for each feature (l1 and l2 norm) to the loss function. This is mainly used to reduce model complexity and address A Nightmare on Machine Learning Street called overfitting.

# In[ ]:


reg = ElasticNet(alpha=1.7)
reg.fit(X_train, y_train) # magic happens here
y_pred = reg.predict(X_val)
y_pred[y_pred < 0] = 0
print('Model MAE:', mean_absolute_error(y_val, y_pred))
print('Mean  MAE:', mean_absolute_error(y_val, np.full(y_val.shape, y.mean())))


# Very good, we achieved better results than the dummy mean prediction. Out final step is to make predictions with the model. For that we use pandas dataframe.

# In[ ]:


# refit and predict submission data
X_train = pipeline.fit_transform(X)
X_test = pipeline.transform(df_test.values)
reg.fit(X_train, y)
y_pred = reg.predict(X_test)
y_pred[y_pred < 0] = 0

df_sub = pd.DataFrame({'Id': np.arange(y_pred.size), 'PRP': y_pred})
df_sub.to_csv('submission.csv', index=False)


# ## Conclusion
# This kernel serves only as an introduction to machine learning with Python. I hope the kernel was useful, leave comments if you have any questions. The model does not guarantee the first place on the leaderboard and there are many things, which can be done next, for example:
#   - multivariate visualization to see patterns in data
#   - apply some transformations to the data
#   - use Binomial Regression - this is often used when the target variable is non-negative and discrete (our case)
#   - better preprocessing
#   - try various models (Decision Trees, Support Vector Machines, etc.)
#   - use cross-validation for model evaluation and hyper-parameters tunning

# 
