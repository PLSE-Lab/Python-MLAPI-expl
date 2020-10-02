#!/usr/bin/env python
# coding: utf-8

# # Wisconsin Breast Cancer 
# 
# This purpose of this kernel is to really get as much feed back as to what I might be doing incorrectly and how I can improve.  This is a work in progress.  My inital goal is to try out some exploratory data analysis stuff, maybe some predictive stuff, and then possibly some interpretive machine learning stuff.  
# 
# Let's see how this goes!

# In[ ]:


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


# Because this is my very first kernel, I'm not going to mess with the cell stuff above!  I'll just add to it below.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')


# In[ ]:


data = pd.read_csv('../input/data.csv')
data.shape


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.info


# So what can I gather from the inital exploratory stuff.  There are 569 rows, 33 columns and no missing data, which is great!  We have this random 'Unamed 32' feature that needs to go away. All features are numerical.  On to univariate exploratory analysis.

# # Univariate Exploring

# In[ ]:


data_id = data.id
diagnosis = data.diagnosis
data.drop(['id', 'Unnamed: 32'], axis = 1, inplace=True)
data.head()


# A side note about me.  I'm coming from the world of R and ggplot, so I love faceting.  So prepare yourselves for lots of faceting!

# In[ ]:


data_long = data.melt(id_vars='diagnosis', var_name='features', value_name='values')
data_long.head()


# ## Values versus Index

# In[ ]:


sns.relplot(data=data_long.reset_index(), x='index', y='values', col='features', col_wrap=3, facet_kws = dict(sharex=False, sharey=False))


# In[ ]:


sns.relplot(data=data_long.reset_index(), x='index', y='values', col='features', col_wrap=3, facet_kws = dict(sharex=False, sharey=False), hue='diagnosis')


# These plots typically help me see if there are any serious outliers at specific indices for a feature.  Overall, these figures hint at the idea that there might be some really great features for helping to separate the **begning** and **malignant** classes of the **diagnosis** target variable.

# ## Histograms and Density plots

# In[ ]:


kwargs = dict(data=data_long, col='features', col_wrap=3, sharex=False, sharey=False)
fg=sns.FacetGrid(**kwargs)
fg.map(sns.distplot, 'values')


# In[ ]:


fg = sns.FacetGrid(**kwargs, hue='diagnosis')
fg.map(sns.distplot, 'values')


# Here I transformed the data into long form and then created histogram and density plots for every numeric feature and then again with respect to the target variable **diagnosis**.  I noticed that some of the density plots appeared bi-modal and by separating the values by **diagnosis**, we can see that some of the features actually do have different distributions with respect to the target variable.  This evidence suggests that some of these features might be really useful in a predictive model while some others could probably be thrown away. For example **fractal_dimension_mean**, **texture_se**, **smoothness_se**, **symmetry_se**, **fractal_dimension_se** provide no clear separation for the two classes of the **diagnosis** target variable.
# 

# ## Boxplots

# In[ ]:


sns.catplot(**kwargs, y='values', kind='box')


# In[ ]:


sns.catplot(**kwargs, x='diagnosis', y='values', kind='box')


# ## Quantile Quantile Plots
# 
# I didn't have any luck finding a quantile plotting method in seaborn or in matplotlib.  I did however find the existence of [scipy.stats.probplot()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html).  So, in the way of python, I'm going to create my own little methods to produce faceted quantile-quantile plots.

# In[ ]:


import scipy.stats as ss

def get_qqplot_data(col):
    (x,y), _ = ss.probplot(col)
    return(pd.DataFrame(dict(x=x, y=y)))

def lmplot_qqplot(data = None, value = None, hue = None, col = None, col_wrap = 3):
    groups = []
    if(col is not None):
        groups.append(col)
    if(hue is not None):
        groups.append(hue)
        
    new_data = data.groupby(groups)[value].apply(get_qqplot_data).reset_index()
    
    sns.lmplot(data=new_data, x='x', y='y', col=col, hue=hue, col_wrap=col_wrap, sharex=False, sharey=False)

    


# In[ ]:


lmplot_qqplot(data = data_long, value='values', col='features')


# In[ ]:


lmplot_qqplot(data=data_long, value='values', hue='diagnosis', col='features')


# ## Normality tests
# 
# I don't even know if these are applicable to this data set, but I'm going to see what happens.  There are a series of tests that I found that can test whether a distribution of values deviate significantly from a normal distribution.  I'm going to implement a small fucntion that can be applied across all the features.

# In[ ]:


def normality_tests(col):
    _, shapiro = ss.shapiro(col)
    _, normaltest = ss.normaltest(col)
    _, skewtest = ss.skewtest(col)
    
    return pd.Series(dict(shapiro = shapiro, normaltest = normaltest, skewtest = skewtest))


# In[ ]:


data_long.groupby(['features']).

