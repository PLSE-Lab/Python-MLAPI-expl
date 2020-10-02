#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import pandas_profiling


# thanks to Nanashi's kernel : https://www.kaggle.com/jesucristo/perfect-eda-in-1-line
# 
# <br>
# ## Check [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)
# Generates profile reports from a pandas `DataFrame`. 
# The pandas `df.describe()` function is great but a little basic for serious exploratory data analysis. 
# `pandas_profiling` extends the pandas DataFrame with `df.profile_report()` for quick data analysis.
# 
# For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
# * **Essentials**: type, unique values, missing values
# * **Quantile statistics** like minimum value, Q1, median, Q3, maximum, range, interquartile range
# * **Descriptive statistics** like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# * **Most frequent values**
# * **Histogram**
# * **Correlations** highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
# * **Missing values** matrix, count, heatmap and dendrogram of missing values
# 
# ## Examples
# 
# The following examples can give you an impression of what the package can do:
# 
# * [NASA Meteorites](http://pandas-profiling.github.io/pandas-profiling/examples/meteorites/meteorites_report.html)
# * [Titanic](http://pandas-profiling.github.io/pandas-profiling/examples/titanic/titanic_report.html)
# * [NZA](http://pandas-profiling.github.io/pandas-profiling/examples/nza/nza_report.html)
# 
# ## Installation
# 
# ### Using pip
# 
# You can install using the pip package manager by running
# 
#     pip install pandas-profiling

# In[ ]:


df = pd.read_csv('../input//titanic/train.csv')

report = pandas_profiling.ProfileReport(df)
report.to_file("report.html")

report


# ### to be continued!
# ### please upvote if you find it usefull!
