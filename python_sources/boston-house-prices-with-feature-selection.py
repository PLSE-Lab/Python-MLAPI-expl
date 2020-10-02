#!/usr/bin/env python
# coding: utf-8

# # Modeling Boston House Prices with Feature Selection
# 

# The Boston House Prices dataset is a great one to get one's feet wet on regression models. The objective is to build the best model to predict median house price based on thirteen features, such as crime rate, accessibility to highways, and property tax rate. This is a classic multivariate regression problem. As you will see later on, the biggest challenge we try to solve when modeling the data is **multicollinearity**, and we will solve it toward the end.

# ## Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Import Dataset

# In[ ]:


# Import the dataset from sklearn library
from sklearn.datasets import load_boston
data = load_boston()
print(data.keys())


# In[ ]:


# Load features, and target variable, combine them into a single dataset
X = pd.DataFrame(data.data, columns=data.feature_names)
# Add constant 
X = sm.add_constant(X)
y = pd.Series(data.target, name='MEDV')
dataset = pd.concat([X, y], axis=1)

# Split training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print dataset description
print(data.DESCR)


# Here we run couple basic functions to make sure the dataset was loaded correctly.

# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# ## Initial Exploration
# Before we implement any algorithms, let's go ahead and explore the dataset.

# First, let's take a look at the distribution of the target variable-Median house price.

# In[ ]:


plt.figure(figsize=(10, 8))
sns.distplot(dataset['MEDV'], rug=True)
plt.show()


# The above graph is a combination of histogram, kernel density estimate, and plot of every single data point. From the graph, we can see that the distribution of the target variable is pretty close to a normal distribution with a few outliers to the right.

# Now let's take a look at the variables and how correlated they are with each other.

# In[ ]:


mask = np.zeros_like(dataset.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(), annot=True, vmin=-1, vmax=1, square=True, mask=mask, cmap=cmap, linewidths=2)
plt.show()


# From the graph above, we can see that there is quite amount of strong and close to strong correlations happening among feature variables, especially between TAX (full-value property-tax rate per $10,000) and RAD (index of accessibility to radial highways). This causes problems when fitting linear models since linear models have assumed no or little multicollinearity.
# 
# We will take care of that as we train our model.

# ## Modeling with Feature Selection

# We will train a multivariate linear regression to model the dataset mapping from features to median house price. Because we have recognized from the last graph that the features are not linearly independent from each other, after every time when we fit a model, we will check the VIF (variance inflation factor) for multicollinearity, and take the feature with the largest off the model. And once every VIF value is lower than 5, we arrive at a model with low enough multicollinearity.
# 
# And it is important to notice that a model built on high multicollinearity dataset, although may fit the current dataset with high accuracy, is not a useful model.
# 
# After implementing the model with constant and without constant, I found there to be a huge difference in the model generated and the R2 values, so I include both implementations down below.

# ### Without Constant X0

# Here we will run through the first iteration of model training, and we will implement a while loop afterwards.

# In[ ]:


# Fit a linear model to data
exog = X_train.drop('const', axis=1)
endog = y_train
model = sm.OLS(endog, exog).fit()


# In[ ]:


# Print model summary
model.summary()


# We can see in warning that there is a strong multicollinearity in the dataset. Let's check the VIF values.

# In[ ]:


# Checking VIF
variables = model.model.exog
vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])
vif


# In this case we will take out the feature with the highest VIF value, which is PTRATIO.

# Now let's write our while loop implementation of our feature selection algorithm.

# In[ ]:


exog = X_train.drop('const', axis=1)
endog = y_train

# Fit model
model = sm.OLS(endog, exog).fit()
# Calculate VIF
variables = model.model.exog
vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])
vif[0] = 0

while sum(np.array(vif) > 5) > 0:
    # Find the id of the feature with largest VIF value
    max_vif_id = np.argmax(vif)

    # Delete that feature from exog dataset
    exog = exog.drop(exog.columns[max_vif_id], axis=1)
    
    # Fit model again
    model = sm.OLS(endog, exog).fit()
    # Calculate VIF
    variables = model.model.exog
    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
    vif[0] = 0

model.summary()


# Now that we arrived at our model with low enough multicollinearity, we can make a more concise model by taking out features with P-value above 0.05 iteratively.

# In[ ]:


# Check if any P-value larger or equal to 0.05
while sum(model.pvalues >= 0.05) > 0:
    # Find the index of the feature of the largest p-value
    max_pvalue_id = np.argmax(model.pvalues)

    # Delete that feature from exog dataset
    exog = exog.drop(max_pvalue_id, axis=1)
    
    # Fit model again
    model = sm.OLS(endog, exog).fit()
    
model.summary()


# After the long process of feature selection, we arrived at our most significant and linearly independent model with four features. The model uses CRIM (per capita crime rate), CHAS (if the area bounds the Charles River), DIS (weighted distances to five Boston employment centres), and RAD (index of accessibility to radial highways) as independent variables to predict MEDV (median house price). The multivariate linear model without any further tuning for polynomial terms achieves an adjusted R-squared score of 0.791 on the training set.

# ### With Constant X0
# 

# Here we will run through the first iteration of model training, and we will implement a while loop afterwards.
# 

# In[ ]:


# Fit a linear model to data
exog = X_train
endog = y_train
model = sm.OLS(endog, exog).fit()


# In[ ]:


# Print model summary
model.summary()


# We can see in warning that there is a strong multicollinearity in the dataset. Let's check the VIF values.

# In[ ]:


# Checking VIF
variables = model.model.exog
vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])
vif


# We can see that the constant has the highest VIF value, which is quite interesting. In this case we will take out the feature with the highest VIF value other than the constant, which is TAX.

# Now let's write our while loop implementation of our feature selection algorithm.

# In[ ]:


exog = X_train
endog = y_train

# Fit model
model = sm.OLS(endog, exog).fit()
# Calculate VIF
variables = model.model.exog
vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])
vif[0] = 0

while sum(np.array(vif) > 5) > 0:
    # Find the id of the feature with largest VIF value
    max_vif_id = np.argmax(vif)

    # Delete that feature from exog dataset
    exog = exog.drop(exog.columns[max_vif_id], axis=1)
    
    # Fit model again
    model = sm.OLS(endog, exog).fit()
    # Calculate VIF
    variables = model.model.exog
    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
    vif[0] = 0

model.summary()


# Now that we arrived at our model with low enough multicollinearity, we can make a more concise model by taking out features with P-value above 0.05 iteratively.

# In[ ]:


# Check if any P-value larger or equal to 0.05
while sum(model.pvalues >= 0.05) > 0:
    # Find the index of the feature of the largest p-value
    max_pvalue_id = np.argmax(model.pvalues)

    # Delete that feature from exog dataset
    exog = exog.drop(max_pvalue_id, axis=1)
    
    # Fit model again
    model = sm.OLS(endog, exog).fit()
    
model.summary()


# When we include the constant term in the model, the features selected are drastically different, and the R2 value is quite different now as well.

# ## Going Forward
# 

# 1. For the two models, we need to test on the test sets for performance.
# 2. Understanding why the constant has such high VIF value and if it's ok to drop the constant
# 3. A decision tree or other non-linear model could be used at modeling this dataset, so we don't need to deal with multicollinearity ourselves.
