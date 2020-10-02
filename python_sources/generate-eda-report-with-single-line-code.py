#!/usr/bin/env python
# coding: utf-8

# ### Introduction

# I found something interesting for us.
# 
# Here, I will tell you how you can generate EDA report of your dataset by using  **pandas-profiling** library
# 
# Basically, It interpretes your Dataframe and calculates following states automatically:
# 
# **Type inference:** detect the types of columns in a dataframe.<br>
# **Essentials:** type, unique values, missing values<br>
# **Quantile statistics** like minimum value, Q1, median, Q3, maximum, range, interquartile range<br>
# **Descriptive statistics** like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness<br>
# **Most frequent values**<br>
# **Histogram**<br>
# **Correlation**s highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices<br>
# **Missing values** matrix, count, heatmap and dendrogram of missing values<br>
# **Text analysis** learn about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data.<br>
# 
# 
# Here, the github link where you can find all setup requirements , dependencies and you can do some modification too.
# 
# <a href ="https://github.com/pandas-profiling/pandas-profiling" target="_blank"> Click Here </a>
# 
# I really appreciate creators and contributors for making this fabulous library.
# 
# 
# Let's get started...

# ### Import *pandas_profiling* package to generate EDA report

# In[ ]:


import pandas as pd 
from pandas_profiling import ProfileReport

#read CSV file 
googleplaystore = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
googleplaystore.head()


# In[ ]:


profile = ProfileReport(googleplaystore, title='Pandas Profiling Report', 
                        html={'style':{'full_width':True}},
                        correlations={"cramers": {"calculate": False}})


# ### Generate Simple EDA report by following code :

# In[ ]:


#profile.to_widgets()


# ### If you want to make your report more attractive and effective then use below code 

# In[ ]:


profile.to_notebook_iframe()


# ### Save your generated report in html file

# In[ ]:


# As a html
profile.to_file(output_file="/kaggle/working/playstore.html")


# ### Save your report in Json format

# In[ ]:


# As a string
json_data = profile.to_json()

# As a file
profile.to_file(output_file="playstore.json")

