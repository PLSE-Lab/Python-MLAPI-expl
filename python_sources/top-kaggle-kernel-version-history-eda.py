#!/usr/bin/env python
# coding: utf-8

# This dataset contains information on 900+  top upvoted kaggle kernels upto 26th February 2018. I wanted to check two basic questions : 
# 
# * ** Does the number of versions a kernel has any impact on popularity?** 
# * **Does the number of days a kernel is worked on has any impact on popularity?**
# 
# Popularity in this dataset can refer to Votes, Forks, Views, Comments etc. I basically saw some users running their kernels multiple times,  so I wanted to know if that has any impact.  Also some kernels in this dataset has been worked over multiple days while other kernels has 2-3 versions at max. However, after extracting the info from the version histories I also checked how the number of kernels ran varies by year and month. (For this I only used the last time the kernel was ran instead of all the other versions, assuming that was the published finalized kernel). 
# 
# Tl, dr answer is neither number of versions of a kernel nor number of days a kernel has been worked on show high positive or negative correlation with Votes, Views, Forks, Comments etc, but since this is a dataset of top kernels, the minimum number of upvotes for a kernel in this dataset is 33(!) and on average the kernels here has 98 upvotes, which is way more than regular 3-4 upvote kernels, if we had checked correlation in the population of all kernels we'd likely get different results. 

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


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


kernels = pd.read_csv("../input/voted-kaggle-kernels.csv")


# In[ ]:


kernels.head()


# In[ ]:


kernels.describe()


# # Extract the version histories

# I've dropped the rows where the version history was null for convinience. We've 953 kernels after dropping the na values. To be clear, the na values does not mean the kernel itself had any issues, rather these refer to some scraping error. 

# In[ ]:


kernels[kernels["Version History"].isnull() == True].head()


# In[ ]:


kernels = kernels.dropna(subset=["Version History"])
versions = kernels["Version History"].str.split("|").dropna()
print(kernels.shape)
print(len(versions))


# In[ ]:


versions[0]


# Then we get the number of versions for each kernel.

# In[ ]:


kernels["number_of_versions"] = versions.map(len)


# The correlation between number_of_versions vs Votes, Comments, Views and Forks is very low and negligible. 

# In[ ]:


sns.heatmap(kernels[["Votes","Comments","Views","Forks","number_of_versions"]].corr(),annot=True)


# To check why we can check the stats for the number of versions a kernel has.  The median is only 11 and we can check from the boxplot that majority of kernels don't have many versions.

# In[ ]:


kernels["number_of_versions"].describe()


# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(y=kernels["number_of_versions"])


# In[ ]:


sns.pairplot(kernels[["Votes","Comments","Views","Forks","number_of_versions","Language"]].dropna(),kind="reg",diag_kind="kde",hue="Language");


# # Checking the Impact Of Outliers

# In[ ]:


def remove_outliers(df):
    from scipy import stats
    df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


# In[ ]:


from scipy import stats
kernels_outliers_removed = kernels[stats.zscore(kernels["number_of_versions"])<3]


# In[ ]:


sns.heatmap(kernels_outliers_removed[["Votes","Comments","Views","Forks","number_of_versions"]].corr(),annot=True)


# Looks like the correlation decreased after removing outliers.

# # Get dates from the version histories
# 
# After that to check how the number of days worked on a kernel impacts popularity, I extracted the number of unique days worked first, because often there's multiple versions of a kernel on the same day. We don't have the exact number of hours a kernel has been worked on, but  unique number of days a kernel has been worked on should be a good proxy.

# In[ ]:


def extract_dates(x):
    temp = [y.split(",")[1] for y in x]
    return temp
     
def extract_unique_dates(dates):
    return pd.to_datetime(dates).unique()


# In[ ]:


kernels["days_worked"] = versions.map(extract_dates)


# In[ ]:


kernels["unique_days_worked"] = kernels["days_worked"].map(extract_unique_dates)
kernels["number_unique_days_worked"] = kernels["unique_days_worked"].map(len)


# In[ ]:


kernels[["number_of_versions","days_worked","unique_days_worked","number_unique_days_worked"]].head()


# Again, the correlation between number of unique days worked and votes, comments etc is negligible. There's a positive correlation between number of versions a kernel has and number of days worked on, as expected.

# In[ ]:


sns.heatmap(kernels[["Votes","Comments","Views","Forks","number_of_versions","number_unique_days_worked"]].corr(),annot=True)


# # Insights From the Dates 

# To extract the dates and months  I've only taken the timestamp of the last version of a kernel instead of all of them, assuming it was the finalized version. 

# In[ ]:


kernels["year"] = kernels["unique_days_worked"].map(lambda x:x[0].year)
kernels["month"] = kernels["unique_days_worked"].map(lambda x:x[0].month)


# In[ ]:


kernels["year"].value_counts().sort_index().plot(kind="bar",figsize=(10,6),color='darkgray')
plt.xlabel("Year")
plt.ylabel("Number of Kernels")


# In[ ]:


months = ['January','February',"March","April","May","June","July","August","September","October","November","December"]
ax = kernels["month"].value_counts().sort_index().plot(kind="bar",color=['r','r']+['darkgray']*8+['r','r'],figsize=(10,6),rot=70)
ax.set_xticklabels(months);
ax.set_xlabel("Months");
ax.set_ylabel("Number of Kernels");
ax.set_title("Kernels Published Monthly Aggregate(2015-2018)")


# Interestingly enough , most of the kernels are last ran between November to February. Personally I think it's because many kagglers get the vacations like Christmas break etc or feel inspired to write kernels in the end of year and beginning of the new year, but loses the momentum in the mid-months. 

# In[ ]:




