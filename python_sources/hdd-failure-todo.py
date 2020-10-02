#!/usr/bin/env python
# coding: utf-8

# # Import the required modules

# In[ ]:


get_ipython().system('pip install git+https://github.com/goolig/dsClass.git')


# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest, SelectFpr
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from dsClass.path_helper import *


# # Load the time-series data and lookt at the features statistics

# In[ ]:


ts_file_path = get_file_path('ts_data.csv')
ts_data = pd.read_csv(ts_file_path)
print(ts_data.shape)
ts_data.head()


# In[ ]:


ts_data.groupby('fail').describe()


# # Plot the time series data

# In[ ]:


original_features = ['bb_count', 'r-w_rate', 'reconsects_count',
       'recovbydrv_count', 'xfer_rate', 'bb_diffs']

fig, axes = plt.subplots(nrows=1, ncols=2)

ts_data.loc[ts_data['d_id']==1,original_features].plot(figsize=(10,12), title='drive '+str(1)+' fail', ax=axes[0])

num_drives = ts_data['d_id'].unique().shape[0]
#num_drives

ts_data.loc[ts_data['d_id']==num_drives,original_features].plot(figsize=(10,12), title='drive '+str(num_drives)+' non-fail', ax=axes[1])
plt.legend()


# # Engineer the time-series features
# 
# For each Create the aggregated features (mean, median, variance, minimum and naximum):
# * for each sn
# * for each feature
# * for each aggregation type
#         aggregate the 20 daily samples into one aggregated sample  

# In[ ]:


#Q1


# # Load the cofiguration data

# In[ ]:


conf_file_path = get_file_path('conf_data.csv')
conf_data = pd.read_csv(conf_file_path)
conf_data.head()


# # Look at the features\labels distribution

# In[ ]:


plt.figure()
conf_data[['age', 'fail']].boxplot(by='fail', figsize=(10,12), sym='')


# In[ ]:


g = conf_data.groupby(["model", "fail"])['d_id'].count().unstack('fail')
g.plot(kind="bar", stacked=True, grid=True, alpha=0.75, rot=45)


# # Handle categorical data 

# For the decision tree algorithm, map the categorical features to numeric with:
# * "Change_capacity" function for "capacity" column
# * "pd.get_dummies" function for "model" column
# 

# In[ ]:


#Q2
def change_capacity(data):
    # A function that receives a data frame and a column name as input and map the categorical capacity feature to numeric 
    di = dict(zip(conf_data["capacity"].unique(), [1000, 600, 300]))

    data.replace({"capacity": di}, inplace=False)


# In[ ]:


new_conf_data = 


# # Merge the aggregated time-series and the configuration datasets

# In[ ]:


#Q3


# # Select the 10 most informative features and transform the data

# In[ ]:


#Q4


# # Train a Decision tree classifier 
# 
# Since we are learning and testing on the same set we will limit the maximum depth parameter to 5 to prevent overfitting

# In[ ]:


#Q5


# # Make the prediction and plot the confusion matrix
# https://en.wikipedia.org/wiki/Confusion_matrix
# 
# * Change "max_depth" paramter to see what happens to model results
# 

# In[ ]:


#Q6

