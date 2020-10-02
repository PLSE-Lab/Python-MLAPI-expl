#!/usr/bin/env python
# coding: utf-8

# # ELO Merchant Category Recommendation
# 
# The goal of this kernel is to get the data in a suitable form for analysis. This is the first kernel in a series which will follow the typical steps in any data science project:
# - get the data (data wrangling)
# - clean the data (data cleaning)
# - explore the data and engineer features (exploratory data analysis)
# - model the data (data modeling)
# - interpret the results and recommend actions
# 
# This is a flexible framework I recommend everyone to follow. In this kernel, we will focus on the first two steps of this analysis.

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Wrangling
# 
# The first step is to get all our data in one place. We use Pandas' functionalities to load our data.

# In[ ]:


# Load the data
data_dict = pd.read_excel('../input/Data_Dictionary.xlsx')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
merchants = pd.read_csv('../input/merchants.csv')
new_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# Then we can examine the first few rows of each dataset in turn to check we have everything in the right format.

# In[ ]:


print(data_dict)

# The Excel documents actually has several sheets, which we load in turn and append
excel_doc = pd.ExcelFile('../input/Data_Dictionary.xlsx')
data_dict_train  = pd.read_excel(excel_doc, 'train', skiprows=2)
data_dict_historical = pd.read_excel(excel_doc, 'history', skiprows=2)
data_dict_new_merchants = pd.read_excel(excel_doc, 'new_merchant_period', skiprows=2)
data_dict_merchant = pd.read_excel(excel_doc, 'merchant', skiprows=2)

data_dict = data_dict_train.append(data_dict_historical)                             .append(data_dict_new_merchants).append(data_dict_merchant)

print(data_dict)


# In[ ]:


historical_transactions.head()


# In[ ]:


merchants.head()


# In[ ]:


new_transactions.head()


# In[ ]:


train_data.head()


# All of the individual datasets have been loaded in their respective dataframes. Let's clean each in turn, before engineering features a way that makes sense for analysis.

# ## Data Cleaning

# We'll go through the dataframes in turn and check for classical examples of dirty data, such as missing values or duplicate rows.

# ### Data Dictionary

# In[ ]:


# Check for null values
data_dict.isnull().sum()


# In[ ]:


# Check for duplicates
data_dict.duplicated().sum()

# Remove duplicates
data_dict.drop_duplicates(inplace=True)


# We remove duplicates which arose from the appending of individual data dictionaries.
# 
# ### Historical Transactions

# In[ ]:


# Check for null values
historical_transactions.isnull().sum()


# In[ ]:


# Let's examine what categories 2 and 3 represent, from the data dictionary
data_dict.loc[(data_dict.Columns == 'category_3') | (data_dict.Columns == 'category_2'),]


# In[ ]:


# Let's examine a few rows with missing merchant ID's
historical_transactions.loc[historical_transactions.merchant_id.isnull(),].head()


# In[ ]:


# Let's drop rows with missing merchant ID's, but keep those with missing categories 2 and 3
historical_transactions.dropna(subset=['merchant_id'], axis=0, inplace=True)


# In[ ]:


# Check for duplicates
historical_transactions.duplicated().sum()


# In[ ]:


# Visualize possible values for categories 1, 2 and 3 to check for invalid data
print(historical_transactions.category_1.unique(),
      historical_transactions.category_2.unique(),
      historical_transactions.category_3.unique())


# In[ ]:


# Definition of purchase amount
print(data_dict.loc[data_dict.Columns == 'purchase_amount',])

# Boxplot
plt.boxplot(historical_transactions.purchase_amount)
plt.show();


# In[ ]:


# Remove 1st outlier at 600,000 (probably was not normalized), and visualize again
historical_transactions = historical_transactions.loc[historical_transactions.purchase_amount < 500000,]
plt.boxplot(historical_transactions.purchase_amount)
plt.show();


# In[ ]:


# Remove more outliers, greater than 2
historical_transactions = historical_transactions.loc[historical_transactions.purchase_amount < 2,]
plt.boxplot(historical_transactions.purchase_amount)
plt.show();


# In[ ]:


# Boxplot for month lag
plt.boxplot(historical_transactions.month_lag)
plt.show();


# In[ ]:


# Boxplot for installments
plt.boxplot(historical_transactions.installments)
plt.show();

# Remove the outlier at 1000 installments
historical_transactions = historical_transactions.loc[historical_transactions.installments < 900,]


# Looking at the missing values, we decide to drop the rows with missing merchant ID's, because it will be impossible for us to match this back to individual merchant information, and hence to compute a loyalty score in later steps. There are missing values in the category_2 and category_3, but we leave them as is, and will create a dummy variable for this later: it is indeed possible for a merchant to only be in one category.
# 
# There are no duplicates, but several rows with outlier values, in particular in the normalized purchase amount columns. My interpretation for the latter is that the amount was not normalized, but lacking the original mean and standard deviation we cannot make the transformation ourselves and hence choose to simply drop these rows.

# ### Merchants

# In[ ]:


# Examine null values
merchants.isnull().sum()


# In[ ]:


# Define the lagged sales
print(data_dict.loc[data_dict.Columns == 'avg_sales_lag3','Description'])

# Drop rows with missing values of lagged sales
merchants.dropna(subset=['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12'], axis=0, inplace=True)


# In[ ]:


# Create boxplots for numerical columns
merchants.boxplot()
plt.xticks(rotation='vertical')
plt.show();


# We keep the rows with missing values in the category_2 column (for a similar reason as the previous dataset) but drop missing values of lagged sales (only 13 rows so it will not make a huge difference). Looking at boxplots for numerical columns we can spot a lot of outliers, but they seem more likely to be valid data than for the previous dataset. Hence we keep these rows for future analysis.

# ### New Transactions

# In[ ]:


# Check for missing values
new_transactions.isnull().sum()


# In[ ]:


# Similar to the historical transactions, we drop missing merchant ID's
new_transactions.dropna(subset = ['merchant_id'], axis=0, inplace=True)


# In[ ]:


# Check for duplicates
new_transactions.duplicated().sum()


# In[ ]:


# Check for outliers
new_transactions.boxplot()
plt.xticks(rotation='vertical')
plt.show();


# In[ ]:


# Similarly to historical transactions, we remove outliers for the purchase amount and installments
new_transactions = new_transactions.loc[(new_transactions.purchase_amount < 2) & (new_transactions.installments < 900),]


# We conduct a very similar data cleaning to historical transactions, removing outliers and missing merchant ID's.

# ### Train

# In[ ]:


# Check for null values
train_data.isnull().sum()


# In[ ]:


# Check for duplicates
train_data.isnull().sum()


# In[ ]:


# Visualize the range of values for the target
plt.boxplot(train_data.target)
plt.show();


# Let's save the data we have cleaned.

# In[ ]:


historical_transactions.to_csv('historical_transactions_clean.csv')


# In[ ]:




