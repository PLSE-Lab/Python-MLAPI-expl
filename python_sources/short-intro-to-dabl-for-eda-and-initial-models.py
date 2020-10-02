#!/usr/bin/env python
# coding: utf-8

# # Intro to dabl Package

# For machine learning tasks, initial phases of data cleaning, pre-processing and analysis take a long time. dabl packge aims to make this process simple and more automated! It is still under development but some of the functions are already super useful, so I recommend using them! This package was created by Andreas Mueller, a core-developer for our well-known ML package Scikit-learn and also a Principal Research SDE at Microsoft.

# The documentation website is as follows:
# 
# https://amueller.github.io/dabl/dev/index.html

# ### Read in Libraries

# In[ ]:


get_ipython().system('pip install dabl')


# In[ ]:


import pandas as pd
import numpy as np
import dabl


# ### Target Variable, categorical features v.s. target, numerical features v.s. target (Regression)

# dabl allows you to do univariate EDA on the target variable and also bivariate EDA on categorical features v.s. target and numerical features v.s. target

# In[ ]:


# Read in the house price dataset
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In this dataset, we want to predict "House Prices", so the feature named 'SalePrice' is our target variable. We want to know how the distribution of the Sales Price looks like and also what kind of relationship different features have with Sales Price.

# In[ ]:


dabl.plot(df, 'SalePrice')


# ### Target Variable, categorical features v.s. target, numerical features v.s. target (Classification)

# In[ ]:


from sklearn.datasets import load_wine
from dabl.utils import data_df_from_bunch

wine_bunch = load_wine()
wine_df = data_df_from_bunch(wine_bunch)

dabl.plot(wine_df, 'target')


# ### Class Distribution

# dabl allows you to examine distributions of different classes of the target variable (in a classification problem)

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from dabl.datasets import load_adult
from dabl.plot import class_hists

data = load_adult()

# histograms of age for each gender
class_hists(data, "age", "gender", legend=True)
# plt.show()


# ### Useful tool for initial model building

# We use the titanic dataset which is already loaded in the dabl package (under .datasets)

# In[ ]:


titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))


# We need to use dabl.clean before we use the "SimpleClassifier" function which will allow us to see which baseline model performs the best

# In[ ]:


titanic_clean = dabl.clean(titanic, verbose=0)


# In[ ]:


fc = dabl.SimpleClassifier(random_state=0) #dabl Simple Classifier

# Divide data into X and y
X = titanic_clean.drop("survived", axis=1)
y = titanic_clean.survived


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfc.fit(X, y)')


# As you can see, it only took 1.55 seconds to try 4 classifiers and measure the common classification metrics (e.g. roc_auc, f1_macro, accuracy etc.) for each of them. You can see which model has the best baseline. This may save you some time as opposed to you having to fit those 4 models separately which will definitely take a longer time.

# Currently, it does not support anything for text or image data, but more functions for these type of data are expected to be on the way. Also, functions for model explanability, hyper parameter tuning etc. are being developed at the moment, so future potential of this package is high!

# ## If you found this kernel helpful, please consider upvoting!
