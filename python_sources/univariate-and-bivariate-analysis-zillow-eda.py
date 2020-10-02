#!/usr/bin/env python
# coding: utf-8

# ## Zillow Exploration
# 
# By: Nick Brooks
# 
# Inspiration:
# - [Feature Selection and Ensemble of 5 Models by Li Yenhsu](https://www.kaggle.com/liyenhsu/feature-selection-and-ensemble-of-5-models/notebook)
# - [Data Visualization in Python: Advanced Functionality in Seaborn by Insight Analytics](https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6)
# - [Zillow EDA On Missing Values & Multicollinearity by Vivek](https://www.kaggle.com/viveksrinivasan/zillow-eda-on-missing-values-multicollinearity)
# 
# Work in Progress..
# 
# ### Part 1
# 1. Univariate Analysis
# 1. Transactions Dataset
#     1. Logerror Distribution
#     1. Transaction Count vs. Mean Log Error
# 1. Properties Dataset
#     1. Mass Histograms
#  
# ### Part 2
#  1. Bi-Variate Analysis
#      1. Scatter Plots
#      1. Correlations

# In[ ]:


# General
import numpy as np
import pandas as pd
import os

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load
properties = pd.read_csv("../input/properties_2016.csv", index_col="parcelid")
transactions = pd.read_csv("../input/train_2016_v2.csv",parse_dates=["transactiondate"])


# # Univariate Analysis
# ## Transaction Dataset
# 
# This dataset includes the property ID, Transaction Date, and the Logarithmic error of the Zillow prediction.

# In[ ]:


transactions.head()


# In[ ]:


# Figure Size
plt.rcParams['figure.figsize'] = (10, 5)
# Density Plot
ax = sns.kdeplot(transactions["logerror"], shade=True)
ax.set_xlabel("Log Error")
ax.set_ylabel("Density")
ax.set_title("Log Error Density")

ax.legend_.remove()
ax.axvline(x=0, color='r', linestyle='-')
plt.show()


# ## Transaction Count vs. Mean Log Error
# 
# This is a play on the Supply and Demand concept, where I wish to see if the transaction frequency has an impact on the predictive accuracy of Zillow's system over different time periods.

# In[ ]:


# Parcing - http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
def mean_count_plot(data, by, title):
    t1= data[["transactiondate","logerror"]].resample(by,on='transactiondate').agg({"mean","count"})
    t1.columns = t1.columns.droplevel()
    ax = t1.plot(secondary_y=["mean"])
    ax.set_title("Transaction Count and Mean by {}".format(title))


# In[ ]:


for x,y in [("7D","Seven Days"),("15D","Fifteen Days"),("M","Month"),("3M","Three Months")]:
    mean_count_plot(transactions, by=x, title=y)


# These two effects appear to be inversly correlated. However, I cannot assume causality because these trends may be seasonal and indepedent. 

# In[ ]:


# By Day in Month
ax = transactions[["logerror","transactiondate"]].groupby(
    [transactions['transactiondate'].dt.day ]).mean().plot()
ax.set_ylabel("Log Error")
ax.set_xlabel("Day in the Month")
ax.legend_.remove()
ax.set_title("Average Log Error by Day in the Month")
plt.show()

# By Weekday
ax = transactions[["logerror","transactiondate"]].groupby(
    [transactions['transactiondate'].dt.weekday ]).mean().plot()
ax.set_ylabel("Log Error")
ax.set_xlabel("Week Day")
ax.legend_.remove()
ax.set_title("Average Log Error by Weekday")
plt.show()


# ## Properties Dataset

# In[ ]:


print("Shape (Heigth, Width): \n{}".format(properties.shape))
print("\nColumn Data Types:")
properties.dtypes.value_counts()


# ## Dealing with Missing Values:
# 
# Me: Hello! Is anybody there! 
# 
# *Data Appears to be Missing :(*

# In[ ]:


missing= (properties.isnull().sum()/properties.shape[0]*100).sort_values(ascending=False).reset_index()
missing.columns = ["Column Name","Percent Missing"]


# In[ ]:


pd.concat([missing[:29], missing[29:].reset_index(drop=True)], axis=1)


# In[ ]:


# Remove Highly Missing Columns
high_miss = missing.loc[missing["Percent Missing"] >34,["Column Name"]]
properties.drop([item for sublist in high_miss.values for item in sublist], axis=1, inplace=True)


# In[ ]:


# Impute
#for col in (x for x in missing["Column Name"] if x is not high_miss):
for col in properties.columns.values:
    if properties[col].dtypes == float:
        properties[col] = properties[col].fillna(properties[col].mean())
        properties[col] = properties[col].astype(int)
    elif properties[col].dtypes == object:
        properties[col] = properties[col].fillna(properties[col].mode().iloc[0])
    elif properties[col].dtypes == int:
        properties[col] = properties[col].fillna(properties[col].median())


# In[ ]:


# Excluding the obj_columns
obj_cols = properties.loc[:,properties.dtypes == object].columns.values
plot_worthy = [x for x in properties.columns.values if x not in obj_cols]

# Number of Unique values by Column
uniqlen = {}
for x in plot_worthy:
    temp = len(properties[x].unique())
    uniqlen[x] = temp

# Columns for Bar Plotting
bar_cols = []
for col,val in uniqlen.items():
    if val < 40: bar_cols.append(col)

# Columns for KDE plotting
hist_cols = [x for x in plot_worthy if x not in bar_cols]


# In[ ]:


print("Missing Values? -> {}".format(properties.isnull().values.any()))
print("\nColumn Data Types:")
properties.dtypes.value_counts()


# ## Mass Histograms
# 
# I was pleasantly surprised to learn that the seaborn FacetGrid map function can handle both seaborn and matplotlib chart types!

# In[ ]:


non_log = ["latitude","longitude","yearbuilt"]
log_hist = [x for x in hist_cols if x not in non_log]


# In[ ]:


grid = sns.FacetGrid(pd.melt(np.log10(properties.loc[:,log_hist]), value_vars=log_hist),
    col="variable",  col_wrap=3 , size=5.0, aspect=0.8, sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()


# In[ ]:


grid = sns.FacetGrid(pd.melt(properties.loc[:,non_log], value_vars=non_log), col="variable",
                     col_wrap=5 , size=4.0, aspect=0.8,sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()


# In[ ]:


# Not Worth Plotting Object Types
print(obj_cols)
for x in obj_cols:
    print(len(properties[x].unique()))
properties.loc[:,obj_cols].sample(5)


# # Bivariate Analysis

# In[ ]:


df = pd.merge(transactions,
   properties.loc[:,log_hist].reset_index(),on="parcelid",how="left")


# In[ ]:


# Log
log_df = pd.merge(transactions, np.log10(properties.loc[:,log_hist]).reset_index(),
                on="parcelid",how="right")
# Melt
melt_log_df = pd.melt(log_df, id_vars="logerror",value_vars=log_hist)

# New Variable to see under/over predictions
melt_log_df["Over/Under"] = melt_log_df.logerror >= 0


# In[ ]:


grid = sns.FacetGrid(melt_log_df, hue="Over/Under"
    ,col="variable",  col_wrap=5 , size=4.0,aspect=0.8,sharex=False, sharey=False)
grid.map(plt.scatter, "value", "logerror")
plt.show()


# In[ ]:


log_df["Over/Under"] = log_df.logerror >= 0
log_df.set_index("parcelid", inplace=True)


# In[ ]:


print("x")


# In[ ]:


# sns.set()
sns.pairplot(log_df.drop(["transactiondate"], axis=1).sample(2000,
             hue='Over/Under',
             diag_kind='kde',
             markers="+",
             # kind="reg",
             size=1.5)
plt.show()


# In[ ]:


plt.savefig("seaborn_pairplot.png")


# In[ ]:


## Explore Correlations

