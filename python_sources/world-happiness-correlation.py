#!/usr/bin/env python
# coding: utf-8

# World Happiness Correlation Chart
# --------------------------------
# using the [2016 World Happiness Report][1] by Sustainable Development Solutions Network
# 
# By Timothy Baney
# 
# 
#   [1]: https://www.kaggle.com/unsdsn/world-happiness

# Step 1 - Import Required Libraries
# ----------------------------------

# In[ ]:


#import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pylab as pylab

get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams[ 'figure.figsize' ] = 10 , 8


# Step 2 - Pull data in as Pandas Dataframes
# ------------------------------------------

# In[ ]:


happiness_2016 = pd.read_csv('../input/2016.csv')


# Step 3 - Add Python function that creates table of correlation coefficients between Pandas dataframe columns, and visually presents them with seaborn
# ------------------------------------------------------------------------

# In[ ]:


def world_corr(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")

world_corr(happiness_2016)


# Voila !
# -------
