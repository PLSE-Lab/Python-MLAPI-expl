#!/usr/bin/env python
# coding: utf-8

# ## *What Makes a Super Spender?*
# _By Nick Brooks, September 2018_
# 
# Building off what I learned from the SRK's [notebook](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue)..

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
import gc
from pandas.io.json import json_normalize

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing

# Load Function by Julian - https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'debug = False\nif debug is True: nrows = 5000\nelse: nrows = None\n\ntrain_df = load_df(nrows = nrows)\ntest_df = load_df("../input/test.csv", nrows = nrows)\n\ndisplay(train_df.sample(4))')


# ## Minority Super-Spenders
# 
# Noticing this stark contrast in customers, I want to explore the data in terms of spenders and non-spenders. My hopes is to capture the bigger picture.

# In[ ]:


train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()

# Distinguish Groups
spending_visitors = gdf.loc[gdf["totals.transactionRevenue"] > 0, "fullVisitorId"]


# ## Pre-Processing

# In[ ]:


# Dropping columns with singular outcome (including NA)
drop_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
print("Columns with one outcome to drop:\nCount: ", len(drop_cols),"\n",drop_cols)
train_df.drop(drop_cols, axis = 1, inplace = True)
test_df.drop([x for x in drop_cols if x in test_df.columns], axis = 1, inplace = True)

# Isolate Dependent Variable and Merge
y_name = 'totals.transactionRevenue'
print("Dependent Variable Name: ", y_name)
y = train_df[y_name]
train_df.drop(y_name, axis= 1, inplace= True)

# Create Train/Test Variable..
train_df["istrain"] = True
test_df["istrain"] = False

# Align Train / Test Columns
notintest = set(train_df.columns).difference(set(test_df.columns))
print("Variables not in test but in train : ", notintest)
train_df.drop(notintest, axis=1, inplace=True)

# Combine Train Test and Remove High Missing..
print("Test and Train columns Match?: ", all(train_df.columns == test_df.columns))
df = pd.concat([train_df, test_df], axis = 0)
del train_df, test_df; gc.collect();

# Remove columns with high missing count
high_miss_cols = [c for c in df.columns if df[c].isnull().sum() > (df.shape[0]*.5)]
print("High Missing Count Columns (50% + Miss):\n", high_miss_cols)
df.drop(high_miss_cols, axis =1, inplace= True)


# ## What Makes a Super Spender? Explore Categorical Variables
# 
# What are the subcategory characteristics of super spenders? This is what these table showcase. Units are **Percentages**. Looking at the **True** columns, the **higher** the percentage, the **more** superspenders in the category.
# 
# Limitations: Class representation is not balanced, therefore this may over-emphasize unusual classes.

# In[ ]:


# Remake Train Set now that processing is done..
train = pd.concat([df.loc[df["istrain"] == True, :], y], axis = 1)
train["spender"] = False
train.loc[train["fullVisitorId"].isin(spending_visitors), "spender"] = True

# Categorical Variables to Explore
categorical_vars = ["channelGrouping", "device.browser", "device.deviceCategory","device.isMobile",
                    "device.operatingSystem","geoNetwork.city","geoNetwork.continent","geoNetwork.country",
                    "geoNetwork.metro","geoNetwork.region","geoNetwork.subContinent"]

# Colored CrossTab
for cols in categorical_vars:
    print("{}".format(cols.title()))
    temp = pd.crosstab(train[cols], train["spender"], dropna=False, normalize = "index").mul(100).round(2)
    temp = temp.sort_values(by=True,ascending= False)[:10]
    display(temp.style.background_gradient(cmap = sns.light_palette("purple", as_cmap=True)))


# ## Deeper Multivariate Interactions

# In[ ]:


train["spender"] = train["spender"].astype("int") # Binary
print("Browser and Grouping")
display(pd.pivot_table(train, values="spender",index="device.browser",columns="channelGrouping").mul(100).round(2).style.background_gradient(cmap = sns.light_palette("red", as_cmap=True)))
print("Device and Grouping")
display(pd.pivot_table(train, values="spender",index="device.deviceCategory",columns="channelGrouping").mul(100).round(2).style.background_gradient(cmap = sns.light_palette("blue", as_cmap=True)))
print("Is Mobile and Grouping")
display(pd.pivot_table(train, values="spender",index="device.isMobile",columns="channelGrouping").mul(100).round(2).style.background_gradient(cmap = sns.light_palette("orange", as_cmap=True)))


# ### Next
# - Explore the continuous variables
