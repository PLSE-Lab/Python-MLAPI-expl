#!/usr/bin/env python
# coding: utf-8

# ![](https://camo.githubusercontent.com/5915a3ee29e2e8be434e69115b247b9dc04d8b09/687474703a2f2f70616e6461732d70726f66696c696e672e6769746875622e696f2f70616e6461732d70726f66696c696e672f646f63732f6173736574732f6c6f676f5f6865616465722e706e67)

# # Simple Profiling using Pandas
# # You can do EDA using this simple method: Pandas Profiling
# 
# Generates profile reports from a pandas DataFrame. The pandas df.describe() function is great but a little basic for serious exploratory data analysis. pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.
# 
# For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
# - **Type inference**: detect the types of columns in a dataframe.
# - **Essentials**: type, unique values, missing values
# - **Quantile statistics** like minimum value, Q1, median, Q3, maximum, range, interquartile range
# - **Descriptive statistics** like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# - **Most frequent values**
# - **Histogram**
# - **Correlations** highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
# - **Missing values matrix**, count, heatmap and dendrogram of missing values
# - **Text analysis** learn about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data.
#  
#  ## You can check the documentation through this link:
# 
# https://github.com/pandas-profiling/pandas-profiling

# To show Pandas Profiling Capabilities, I preferred a simple dataset : Iris Species
# 
# ![](https://thegoodpython.com/assets/images/iris-species.png)
# 
# You can download it from https://www.kaggle.com/uciml/iris or you can also find it on the UCI Machine Learning Repository 
# ( http://archive.ics.uci.edu/ml/index.php )
# 
# The dataset includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# 
# - Id
# - SepalLengthCm
# - SepalWidthCm
# - PetalLengthCm
# - PetalWidthCm
# - Species
# 
# ![](https://www.kaggle.io/svf/138327/e401fb2cc596451b1e4d025aaacda95f/sepalWidthvsLength.png)
# 
# notes are taken from https://www.kaggle.com/uciml/iris

# 
# 
# 

# # Import the libraries

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


print(os.listdir("../input"))


# # Import the dataset

# In[ ]:


df_iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


df_iris.info()


# In[ ]:


df_iris.describe()


# # Bam! Magic!

# In[ ]:


pp.ProfileReport(df_iris)


# # I try to update it frequently , thank you very much for checking my notebook!
