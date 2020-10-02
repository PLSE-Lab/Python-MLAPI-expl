#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# This notebook is a template to give directions on our expectations and serve as a proxy for a rubric as well. 
# You will find a copy of this in the Vocareum workspace as well. If you do decide to use notebooks on kaggle, 
# do keep your work private, or just share with your team mates only. 


# If you have any packages or libraries that you would like to pre-load, put them here (next cell)

# In[ ]:


# Packages and libraries load here [basic packages are specified; additional packages may be needed]
get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
import numpy as np

import missingno as msno
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

import matplotlib.pylab as plt

#from dmba import regressionSummary, adjusted_r2_score, AIC_score, BIC_score


# In[ ]:


# Load Data set here
# Input data files are available in the "../input/" directory.
# For example, running this cell(by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Perform initial inspection of the data (add as many cells as you need).
# Aspects may include:
# 
# 1. Cursory look at the rows and columns
# 2. Inspect the datatypes of the attributes and response
# 3. Counts of the observations and variables
# 4. Summary statistics of each variable
# 5. Discussion or Insights from the exploration above

# In[ ]:


# Initial inspection begins here


# ## Perform data visualization (add as many cells as you need). Aspects may include:
# 
# 1. Scatter plots for at least four relevant bivariate relationships.
# 2. Distribution plots for at least two relevant numeric variables
# 3. Box plots for at least two numerical-categorical relationships. 
# 4. Bar charts for at least two interesting relationships.
# 5. Include discussions why you selected the plots/charts along with the outputs 
# 6. Share insights that you gain from these plots, in context of the value to a business.[](http://)

# In[ ]:


# Data Visualization begins here


# ## Visualize the Pairwise Correlations and comment below
# 

# In[ ]:


# Visualize pairwise correlations and comment here


# ## Check for missing values and comment below

# In[ ]:


#Check for missing values here and find ways to handle them.


# ## Inspect categorical variables here (add as many cells as you need) 
# Aspects may include:
# 1. Counts of each subtype of categorical variables
# 2. Analysis of each categorical variable as nominal or ordinal
# 3. Discussion or comments for each of the above

# In[ ]:


# Inspect categorical variables here


# # The sections above were all about Exploratory Data analysis. These were to be done in teams
# # The following sections should be done independently. You can copy the notebooks and create one for each team member

# ## Create X and y objects to hold the predictors and response variable, respectively

# In[ ]:


# Establish X (predictors) and y (response)


# ## Encode categorical variables here 
# Aspects may include:
# 1. Dummy coding (one-hot encoding) nominal variables
# 2. Label encoding for ordinal variables. This is already done in the dataset,but do verify this.  

# In[ ]:


#Encode Categorical variables


# ## Split the data into training and validation subsets 
# (optional; used to quickly test models with a small validation set of no more than 10% of the full dataset)
# ### NOTE: The project_Leaderboard dataset is used to generate response estimates for submitting to Kaggle.

# In[ ]:


# Split the data here.  


# ## Pre-model data processing
# fit_transform on the training data, transform on the test data to maintain information in-
# tegrity of the test data Aspects may include:
# 1. Standardization or normalization of the predictor values
# 2. Transformations of predictors or response (eg., Box-Cox, log, square root) 
# 3. Using a variance threshold to feature select predictors
# 4. Discussion or comments for each of the above

# In[ ]:


# data processing here 


# ## Algorithm definition and model building here
# Build at least three models. Justify the model you select

# In[ ]:


# Model building here 


# ## Generate performance information and analyze the results

# In[ ]:


# Performance Analysis 

