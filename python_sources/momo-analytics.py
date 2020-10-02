#!/usr/bin/env python
# coding: utf-8

# # MOMO ANALYTICS
# ## Author: Vu Duong
# ### Date: July, 07, 2020

# # INTRODUCTION
# The aim of the campaign is to know if the different MoMo ages have different serviceid cluster, and how we do cross-sale to customers
# 
# ## Data Description
# - Servicedid: The identity of the transaction
# - Date: the date that transaction is performed
# - User_id: The identity of the individual customer
# - Momo_age: the difference between LastServiceDate and FristServiceDate

# ## Result from the study
# 1.We would like to know whether users in different MoMo ages have different serviceid cluster?
# - Customer in 100_Momo_age group tend to use Serviceid of 667, 333, 1014, 487, 981
# - Customer in 200_Momo_age group tend to use Serviceid of 981, 20, 1014, 271, 666 
# - Customer in 300_Momo_age group tend to use Serviceid of 18, 333, 667, 981, 268
# - Customer in 300_plus_Momo_age group tend to use Serviceid of 981, 19, 271, 1014, 2
# 
# 2.We would like to know which serviceid we can cross-sales to users?
# - If a customer converts Serviceid_333, we suggest them the Serviceid_667 with Support of 20%, Confidence of 100%, Lift being greater than 1.
# - If a customer converts Serviceid_667, we suggest them the Serviceid_333 with Support of 20%, Confidence of 91.8%, Lift being greater than 1.
# - If a customer converts Serviceid_981, we suggest them the Serviceid_1014 with Support of 16.9%, Confidence of 61%, Lift being greater than 1.
# - Do the same procedure for the rest of the table

# # LIBRARY

# In[ ]:


# Data Processing
import numpy as np
import pandas as pd
import datetime as dt
import functools

# Data Visualizing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display, HTML
from IPython.display import Image

# Data Clustering
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion

# Data Modeling
from sklearn.ensemble import RandomForestRegressor

# Math
from scipy import stats  # Computing the t and p values using scipy 
from statsmodels.stats import weightstats 

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# # DATA EXPLORATION

# In[ ]:


df1 = pd.read_csv('../input/momo-data-process/data process.csv')
df2 = pd.read_csv('../input/momo-analytics1/Momo_Analytics.csv')


# In[ ]:


df1.rename(columns = {'user_id':'User_id'}, inplace=True)
df1.head()


# In[ ]:


df2.head()


# # ANALYSIS 1

# ## 1.We would like to know whether users in different MoMo ages  have different serviceid cluster?

# In[ ]:


dfs = [df1, df2]
CustomerTable = functools.reduce(lambda left,right: pd.merge(left,right,on='User_id', how='outer'), dfs)
CustomerTable.dropna(inplace=True)


# In[ ]:


CustomerTable.head()


# In[ ]:


CustomerTable['firstservicedate'] = pd.to_datetime(CustomerTable['firstservicedate'])
CustomerTable['secondservicedate'] = pd.to_datetime(CustomerTable['secondservicedate'])
CustomerTable['lastservicedate'] = pd.to_datetime(CustomerTable['lastservicedate'])
CustomerTable['Date'] = pd.to_datetime(CustomerTable['Date'])


# In[ ]:


CustomerTable.head()


# In[ ]:


CustomerTable['Momo_age'] = 1 + (CustomerTable['lastservicedate']-CustomerTable['firstservicedate']).dt.days


# In[ ]:


# Split customers into 4 categories of 100_Momo_age, 200_Momo_age, 300_Momo_age, 300_plus_Momo_age

bins = [-1, 100, 200, 300, np.inf]
names = ['100_Momo_age', '200_Momo_age', '300_Momo_age', '300_plus_Momo_age']
CustomerTable['Momo_age_bin'] = pd.cut(CustomerTable['Momo_age'], bins, labels=names)


# In[ ]:


CustomerTable.head()


# In[ ]:


CustomerTable.groupby(['Momo_age_bin','Serviceid'])[['User_id']].count().dropna()


# #### Observation
# - Customer in 100_Momo_age group tend to use Serviceid of 667, 333, 1014, 487, 981
# - Customer in 200_Momo_age group tend to use Serviceid of 981, 20, 1014, 271, 666 
# - Customer in 300_Momo_age group tend to use Serviceid of 18, 333, 667, 981, 268
# - Customer in 300_plus_Momo_age group tend to use Serviceid of 981, 19, 271, 1014, 2

# In[ ]:


# Grouping customers bases on categories of Momo_age_bin and servicedid, 
# in order to know which services were used most by each segment
CustomerAgg = CustomerTable.groupby(['Momo_age_bin','Serviceid'])[['User_id']].count().dropna()

# Source: https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
# We group by the first level of the index:
g = CustomerAgg['User_id'].groupby(level=0, group_keys=False)

# Then we want to sort 'order' each group and take the first five elements:
# Alternative:  res = g.apply(lambda x: x.order(ascending=False).head(5))
g.nlargest(5)


# ## 2. We would like to know which serviceid we can cross-sales to users?

# In[ ]:


# CrossSale: https://www.youtube.com/watch?v=VMavY0pBo2o
CustomerTable.head()


# In[ ]:


Cross_sale = CustomerTable[['User_id', 'Date', 'Serviceid']]
Cross_sale['User_id & Date'] = Cross_sale['User_id'].astype('str') + ' ' + Cross_sale['Date'].astype('str')
Cross_sale.drop(['User_id', 'Date'], axis=1, inplace=True)


# In[ ]:


# To eliminate duplicate rows. For example, a customer use 1 particular serviceid twice a day, 
# all we need is to know the number of unique serviceids per day for each customer.
# Another example, a customer usually buy 2 toothbrushes along with 1 toothpaste. Thus, we can cross-sell 2-item combo.
Cross_sale.drop_duplicates(inplace = True)


# In[ ]:


Cross_sale.set_index('User_id & Date', inplace=True)


# In[ ]:


Cross_sale.head(10)


# In[ ]:


Cross_sale['Serviceid'] = Cross_sale['Serviceid'].astype('str')

basket = pd.get_dummies(Cross_sale)
basket.head()


# In[ ]:


basket_sets = pd.pivot_table(basket, index='User_id & Date', aggfunc='sum')
basket_sets


# In[ ]:


frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[ ]:


# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02)]


# ### Observations:
# - If a customer converts Serviceid_333, we suggest them the Serviceid_667 with Support of 20%, Confidence of 100%, Lift being greater than 1.
# - If a customer converts Serviceid_667, we suggest them the Serviceid_333 with Support of 20%, Confidence of 91.8%, Lift being greater than 1.
# - If a customer converts Serviceid_981, we suggest them the Serviceid_1014 with Support of 16.9%, Confidence of 61%, Lift being greater than 1.
# - Do the same with the following rows.
# - Further explaination: https://towardsdatascience.com/association-rules-2-aa9a77241654

# In[ ]:


# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()


# In[ ]:


# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();


# In[ ]:




