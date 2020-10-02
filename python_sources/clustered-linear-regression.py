#!/usr/bin/env python
# coding: utf-8

# # Clustered Linear Regression
# 
# For a linear regression approach we try to fit a best model on entire dataset. 
# However often we have seen within dataset based on a particular feature the dataset behaves totally different and single model is not the best solutions, instead have multiple model which applied on different subset or filtered data does better.
# 
# The library/package can be found on [pypi.org here](https://pypi.org/project/kesh-utils/) and source code on [github here](https://github.com/KeshavShetty/ds/tree/master/KUtils/linear_regression)

# # How to find the feature which splits the dataset into multiple sub dataset (and there after build and apply different models)
# There is no easy solution, instead use trial and error or brute force to subset data on different feature and build multiple model.
# This clustred or grouped Linear Regression does the same.
# You send the entire dataset and specifiy list of columns to separate the dataset individually and return the kpi measures like rmse or r2 etc and then decide which way to go.

# # How "Clustered Linear Regression" works?
# - First it lists possible combinations
# - For each possible combinations split the data into subset
# - For each subset execute the Auto Linear Regression. Check previous [kaggle post](https://www.kaggle.com/keshavshetty/auto-linear-regression) on this.
# - Return summary or consolidated kpi measures at group level.

# # Action time
# 
# Lets try this library and see how it works
# 
# To demonstrate the library I used the one of the popular dataset [UCI Diamond dataset](https://www.kaggle.com/shivam2503/diamonds) from Kaggle
# 

# In[1]:


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


# In[2]:


# Install Clustered Linear Regression (Part of KUtils Package)
get_ipython().system('pip install kesh-utils')


# In[3]:


# Load the custom packages from kesh-utils
from KUtils.eda import chartil
from KUtils.eda import data_preparation as dp
from KUtils.linear_regression import auto_linear_regression as autolr
from KUtils.linear_regression import clustered_linear_regression as clustlr


# In[5]:


# Some warning from pandas and Numpy need to ignore for time being (Some like conversion from int to float, cyclyic subset etc)
import warnings  
warnings.filterwarnings('ignore')


# In[6]:


# Use 3 decimal places for decimal number (to avoid displaying as exponential format)
pd.options.display.float_format = '{:,.3f}'.format


# In[7]:


# Load the dataset
diamond_df = pd.read_csv('../input/diamonds.csv')


# In[8]:


# Have a quick look on the top few records of the dataset 
diamond_df.head()


# In[9]:


diamond_df.describe()


# In[10]:


# Drop first column which is just a sequence
diamond_df = diamond_df.drop(diamond_df.columns[0], axis=1)


# In[12]:


diamond_df['price'] = diamond_df['price'].astype(float) # One of the warning can be escaped


# In[13]:


diamond_df.head()


# # First check the Single model on entire dataset and see the best model and its performance 
# We will use Auto Linear Regression and check the result. Refer previous [kaggle post](https://www.kaggle.com/keshavshetty/auto-linear-regression) on this

# In[15]:


# Auto Linear Regression - Single model for entire dataset
model_info = autolr.fit(diamond_df, 'price', 
                     scale_numerical=True, acceptable_r2_change = 0.005,
                     include_target_column_from_scaling=True, 
                     dummies_creation_drop_column_preference='dropMin',
                     random_state_to_use=44, include_data_in_return=True, verbose=True)


# In[17]:


# model_iteration_info
model_info['model_iteration_info'].head()


# # Perfromance of single model
# We can clearly see the final model has RSquare=0.913 and RMSE=0.295 with 20 features

# # Now Clustered Linear regression
# Now lets try clustred or grouped Linear regression. For this we need to send list of columns as dataset filtering parameter.
# In this case we will use [ 'cut', 'color','clarity' ] which are categorical features. 
# 
# In this demo it will create 3 groups 'cut', 'color','clarity'.
# Within eachgroup data is further divided or filtered based on number of unique level or label in that respective categorcial feature.
# 
# *We can use continuous variable as well which gets auto 10 bins created. For now we will use only categorical columns
# 
# # The API clustlr.fit() has below parameters
# - data_df (Full dataset)
# - feature_group_list (List of column on which filter and group the data
# - dependent_column (The target column)
# - max_level = 2 (When it is 2 it uses two feature combination to filter)
# - min_leaf_in_filtered_dataset=1000 (Condition the minimum datapoints in subgroup without which autolr will not be executed)
# - no_of_bins_for_continuous_feature=10 (number of bins to be created when you use continuous varibale for grouping)
# - verbose (Use True if you want detailed debug/log message)

# In[18]:


group_model_info, group_model_summary = clustlr.fit(diamond_df, feature_group_list=['cut', 'color','clarity'], dependent_column='price', 
                                                    max_level = 1, min_leaf_in_filtered_dataset=500,
                                                    verbose=True)


# ## The function Clustered Linear Regression fit() returns group_model_info, group_model_summary
# - group_model_info - This contains each subgroup/subset level model built and its performance.
# - group_model_summary - This contains group level consolidated measurements.**

# In[19]:


# Check the modle summary
group_model_summary


# ### We can see that splitting dataset by feature 'clarity' and applying different models will gives the best return of R2=0.943.
# Other values are 
# 
# - By 'cut' - RMSE=0.286 & R2=0.918
# - By 'color' - RMSE=0.314 & R2=0.897
# - By 'clarity' - RMSE=0.237 & R2=0.943

# In[20]:


# Check subgroup level model efficieny and the dataset size used for each subset
group_model_info


# ## Conclusion: Splitting the dataset on feature'clarity' and building seperate model for each subset will give much better result than a single model.
# 
# The perfromance of single model was RSquare=0.913 and RMSE=0.295
# 
# Whereas multiple model by splittin dataset on feature 'clarity' has mean performance RMSE=0.237 & R2=0.943.
# 
# #### * <font color='red'>Caution: When you filter dataset the new subset will have reduced population which may not be sufficient or suitable to build the model. Use the parameter "min_leaf_in_filtered_dataset" control or condition the subset size.</font>
# 

# In[25]:


# Do some visualization and analysis on feature 'clarity' and see why it performs better when it is split using that feature
chartil.plot(diamond_df, ['clarity', 'price'], chart_type='violinplot')


# #### So it clearly shows there is wide distribution of price among different subgroup of feature clarifty.
# 
# Do further analysis on the feature subset.

# ## Dropping feature may not always lose information. In fact splitting dataset on feature and using multiple model may have better result. This is mainly caused by differnt behaviour amoung the subset data.
# 
# Please explore further to understand how the library works and suits for your dataset.

# In[ ]:




