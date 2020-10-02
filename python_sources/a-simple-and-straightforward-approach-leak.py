#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# - Predict the value of transaction for potential customers
# - This is a <b>Regression</b> problem
# - URL of the problem: https://www.kaggle.com/c/santander-value-prediction-challenge

# ### Importing Libraries

# In[ ]:


# Linear algebra
import numpy as np  

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# For building a model
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import warnings
warnings.filterwarnings('ignore')


# ### Loading the data

# In[ ]:


santander_df = pd.read_csv('../input/train.csv')


# <b>There are no null values in the dataset</b>

# ### Miscellaneous

# In[ ]:


santander_df.head()


# <b>There are 12 important features in this data</b><br>
# <b>We'll use SelectKBest to select those 12 important features</b>

# ### Splitting the data

# In[ ]:


X = santander_df.drop(['ID', 'target'], axis=1)
y = santander_df.target


# In[ ]:


chi_select = SelectKBest(score_func=chi2, k=12)


# <b>Before applying the fit method, all the labels in the target variable must be of the same dtype</b>

# In[ ]:


y = np.array(y).astype('int')


# In[ ]:


y.dtype


# In[ ]:


chi_select.fit(X, y)


# In[ ]:


chi_support = chi_select.get_support() # Contains the values either True or False, True means the feature has been
                                       # selected


# In[ ]:


chi_features = X.loc[:, chi_support].columns.tolist() # Storing the selected features


# ### 12 Important Features

# In[ ]:


chi_features


# In[ ]:


X = santander_df[chi_features]  # Limiting our X to only 12 selected features


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=6)


# ### Building the model

# In[ ]:


reg_tree = DecisionTreeRegressor(random_state=2) # No HyperParameters


# In[ ]:


reg_tree.fit(X_train, y_train)  # Training the model


# In[ ]:


predictions = reg_tree.predict(X_test) # Predicting on the unseen data


# <b>Making the dtype of predictions as int</b>

# In[ ]:


predictions = np.array(predictions).astype('int')


# In[ ]:


mse = metrics.mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# ### Submission

# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X = test_df[chi_features]


# In[ ]:


ID = test_df.ID


# In[ ]:


target = reg_tree.predict(X)


# In[ ]:


target


# In[ ]:


submit_df = pd.DataFrame()


# In[ ]:


submit_df['ID'] = ID


# In[ ]:


submit_df['target'] = target


# In[ ]:


submit_df.head()


# In[ ]:


submit_df.to_csv('submission.csv')

