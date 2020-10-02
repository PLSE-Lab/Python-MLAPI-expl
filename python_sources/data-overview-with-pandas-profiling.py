#!/usr/bin/env python
# coding: utf-8

# # EDA pandas-profiling
# pandas-profitting generates profile reports from a pandas DataFrame. You can do exploratory data analysis via df.describe() but that is a little bit basic. You can see each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
# - Essentials: type, unique values, missing values
# - Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
# - Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# - Most frequent values
# - Histogram
#  -Correlations highlighting of highly correlated variables, Spearman and Pearson matrixes
# 
# See detail in https://github.com/pandas-profiling/pandas-profiling
# 

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pdp


# In[ ]:


X_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
X_test = pd.read_csv('../input/X_test.csv')


# In[ ]:


profile_X_train = pdp.ProfileReport(X_train)
profile_X_train


# In[ ]:


profile_y_train = pdp.ProfileReport(y_train)
profile_y_train


# In[ ]:


profile_X_test = pdp.ProfileReport(X_test)
profile_X_test


# In[ ]:


profile_X_train.to_file(outputfile="X_train.html")
profile_y_train.to_file(outputfile="y_train.html")
profile_X_test.to_file(outputfile="X_test.html")

