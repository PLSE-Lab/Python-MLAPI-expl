#!/usr/bin/env python
# coding: utf-8

# <h1>Accelerated Data Analytics using Pandas-Profiling</h1>

# # Introduction
# 
# 
# [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) generates for each column the following statistics - if relevant for the column type:
# 
# * **Essentials**: type, unique values, missing values
# * **Quantile statistics** like minimum value, Q1, median, Q3, maximum, range, interquartile range
# * **Descriptive statistics** like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# * **Most frequent values**
# * **Histogram**
# * **Correlations** highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
# * **Missing values** matrix, count, heatmap and dendrogram of missing values   
# 
# 
# Use of the library is extremely simple, with one single line of code. If your dataset is `df`, you will visualize the report with the following command:
# 
# 
# > pandas_profiling.ProfileReport(df)
# 
# Source code, installation instructions, documentation and usage examples <a href="https://github.com/pandas-profiling/pandas-profiling">here</a>.  
# 
# A nice article on Medium reviewing the features, <a href="https://blog.usejournal.com/pandas-profiling-to-boost-exploratory-data-analysis-8e718238bcd1">here</a>.  
# 
# An interesting article of useful Machine Learning tools like Pandas Profiling, <a href="https://fizzylogic.nl/2018/08/21/5-must-have-tools-if-youre-serious-about-machine-learning/">here</a>.

# # Prepare for data analysis
# 
# ## Load packages

# In[ ]:


import pandas as pd
import pandas_profiling


# ## Load data   
# 

# In[ ]:


data_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# # Data exploration
# 
# We use for data exploration [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling).

# In[ ]:


pandas_profiling.ProfileReport(data_df)

