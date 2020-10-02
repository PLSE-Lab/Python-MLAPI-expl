#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) for Categorical Variables - 2 functions is all you need

# [In previous Kernel](https://www.kaggle.com/nextbigwhat/eda-for-categorical-variables-a-beginner-s-way) we looked at step by step how to perform very basic 2 EDAs which are essential to get your hands on the data and to know just enough when you are starting off with the competition. 
# 
# 1. Basic Statistics <br>
# 2. Distribution Plots <br>
#  2.1 Frequency distribution for each Independent Variable <br>
#  2.2 Relationship between the Dependent Variable & Inependent Variables
#  
#  In this kernel we will put all the individual components together 
# 

# ## Notebook Content
# 
# 1. [Basic Statistics](#s1) <br>
# 
# 2. [Frequency Distribution](#s2) <br>
# 
# 3. [Conclusion](#s3) <br>

# In[ ]:


# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')

# Comment this if the data visualisations doesn't work on your side
get_ipython().run_line_magic('matplotlib', 'inline')

# We are using whitegrid style for our seaborn plots. This is like the most basic one
sns.set_style(style = 'whitegrid')


# In[ ]:


dataset = pd.read_csv('../input/train.csv')


# In[ ]:


nrow, ncol = dataset.shape
nrow, ncol


# Our Data has 1460 Rows and 81 Columns

# # EDA for Categorical Variables 
# 
# #### Create a separate dataframe which has only Categorical Variables

# In[ ]:


ds_cat = dataset.select_dtypes(include = 'object').copy()
ds_cat.head(2)


# <a id='s1' >
# ## 1. Basic Statistics</a>

# In[ ]:


ds_cat_stats = pd.DataFrame(columns = ['column', 'values', 'values_count_incna', 'values_count_nona', 
                                       'num_miss', 'pct_miss'])
tmp = pd.DataFrame()

for c in ds_cat.columns:
    tmp['column'] = [c]
    tmp['values'] = [ds_cat[c].unique()]
    tmp['values_count_incna'] = len(list(ds_cat[c].unique()))
    tmp['values_count_nona'] = int(ds_cat[c].nunique())
    tmp['num_miss'] = ds_cat[c].isnull().sum()
    tmp['pct_miss'] = (ds_cat[c].isnull().sum()/ len(ds_cat)).round(3)*100
    ds_cat_stats = ds_cat_stats.append(tmp)
    
ds_cat_stats


# In[ ]:


# Let's do an Ascending sort on the Numboer of Distinct Categories for each categorical Variables
ds_cat_stats.sort_values(by = 'values_count_incna', inplace = True, ascending = True)

# And set the index to Column Names
ds_cat_stats.set_index('column', inplace = True)
ds_cat_stats


# The idea here is to familiarize ourself with data just enough. And also to make mental notes
# 
# **Quick Observations**
# 1. look at the Variable 'Alley' -- distinct categories are 3 but without nans it is 2. [Mental Note :  label encoding or one-hot encoding]
# 2. Last 3 variables - Exteriror1st, Exteriror2nd, Neighborhood have more than 10 categories. [Mental Note : We should keep this in mind while doing the dummy variable coding]
# 3. 5 variables have more than 47% missing values. 

# In[ ]:


ds_cat_stats.sort_values(by = 'pct_miss', ascending = False).head(5)


# ---
# <a id = 's2'>
# ## 2. Frequency Distribution </a>
# Here again we will start with one variable (MSZoning) as we did in part 1 to build our code and then subsequently we will put everything together 
# 
# #### Since we are working on a supervised ML problem we should also look at the relationshipt between the dependent variable and independent variable. In order to do that let's add our dependent variable to this dataset.

# In[ ]:


ds_cat['SalePrice'] = dataset.loc[ds_cat.index, 'SalePrice'].copy()


# In[ ]:


ix = 1
fig = plt.figure(figsize = (15,10))
for c in list(ds_cat.columns):
    if ix <= 3:
        if c != 'SalePrice':
            ax1 = fig.add_subplot(2,3,ix)
            sns.countplot(data = ds_cat, x=c, ax = ax1)
            ax2 = fig.add_subplot(2,3,ix+3)
            sns.boxplot(data=ds_cat, x=c, y='SalePrice', ax=ax2)
            #sns.violinplot(data=ds_cat, x=c, y='SalePrice', ax=ax2)
            #sns.swarmplot(data = ds_cat, x=c, y ='SalePrice', color = 'k', alpha = 0.4, ax=ax2)
            
    ix = ix +1
    if ix == 4: 
        fig = plt.figure(figsize = (15,10))
        ix =1


# **Observations**
# 
# 1. Since Exterior1st, Exterior2nd, Neighborhood have lot more distinct categories, we need to look at them separately
# 2. Some of the variables with binary categories are heavily skewed, for ex look at the variable Street, Utilities, Central Air
# 3. Variables that have more than 2 categories also show skewed patterns, for ex look at the variables Condition1, Condition2, BldgType, BldgCond etc
# 4. Observe the variables GarageQual, GarageCond heavily skewed again. We don't know what TA means we will refer to the Data Dictionary
# 
# Remember: These are for mental notes and we are not deeply investigating any of these variables yet
# 
# Next step is to do Data Processing of these categorical variables. Which we will go into later
# 
# Also, We will look at the EDA for Numerical Variables in a separate Kernel

# <a id = 's3' >
# ## Conclusion</a>
# 
# That's it for this kernel. 
# 
# Idea of this kernel is just to combine our learnings from previous kernel and show in a short and clear way how effectively you can do EDA for categorical variables
# 
# We definitely don't need to put too much weight on the insights that can be gained from an EDA like this. At most we get 2-dimensional relationships, which can be misleading. We will rather focus more on Machine Learning driven EDA.
# 
# But the quest is not over yet! 
# 
# On the basis of the feedbacks/suggestions that I recieve for the kernel, I would add them and convert into a different Kernel so that we can see the progression for our learnings.

# In[ ]:





# In[ ]:





# In[ ]:




