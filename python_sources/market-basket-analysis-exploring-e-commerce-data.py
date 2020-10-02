#!/usr/bin/env python
# coding: utf-8

# # **Market Basket Analysis - Exploring eCommerce data**
# 
# **This Kernel was used as a presentation material for the Intertalent Conference at University of Debrecen , Hungary 2018**
# 
# **Introduction**
# 
# My first experience with Market Basket Analysis (MBA) projects was in Brazil, deploying this particular solution to a Retail tech company focused on improving marketing performance working data-driven. By that time my analytical team had good knowledge of statistics but no Python practice at all, therefore, in order to keep with the project we worked entirely on [RapidMiner](http://rapidminer.com). This platform has improved a lot since our first experiments, and even though many people would prefer to jump into the code learning curve, I still strongly suggest starting with code-free software like RapidMiner for getting a sense of how Data Science could work for you and your company.
# This Kernel is basically a similar python implementation of the same technology used with some extra time series analysis. Association rules are powerful for marketing and could be an initial source for recommendation systems.
# 
# This article from KDNuggets is a great place to start in case you've never seen MBA Implementations and statistics involved: https://bit.ly/2qzxh8H 
# 
# **This notebook is structured as follows:**
# 
#     1. Loading libraries and data
#     2. Handling missing data with missingno library
#           2.1 Data Loss Management (DLM)
#     3. Data visualization
#     4. Frequent sets and association rules with apriori
#     5. Conclusions
#    

# # 1. Loading libraries and data:

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Loading libraries for python
import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Advanced data visualization
import re # Regular expressions for advanced string selection
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion
from mlxtend.preprocessing import OnehotTransactions # Transforming dataframe for apriori
import missingno as msno # Advanced missing values handling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading input, converting InvoiceDate to TimeStamp, and setting index: df
# Note that we're working only with 30000 rows of data for a methodology concept proof 
df = pd.read_csv('../input/data.csv', nrows=30000)
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
df.set_index(['InvoiceDate'] , inplace=True)

# Dropping StockCode to reduce data dimension
# Checking df.sample() for quick evaluation of entries
df.drop('StockCode', axis=1, inplace=True)
df.sample(5, random_state=42)


# # 2. Handling missing data with missingno library:

# In[3]:


# Checking missing and data types
# Experimenting missingno library for getting deeper understanding of missing values 
# Checking functions on msno with dir(msno)
print(df.info())
msno.bar(df);
msno.heatmap(df);

# InvoiceNo should be int64, there must be something wrong on this variable
# When trying to use df.InvoiceNo.astype('int64') we receive an error 
# stating that it's not possible to convert str into int, meaning wrong entries in the data.


# In[4]:


# Zoming into missing values
# On df.head() only CustomerID is missing
# We notice the same problem in Description when exploring find_nans a bit
find_nans = lambda df: df[df.isnull().any(axis=1)]


# ## 2.1 Data Loss Management (DLM)

# In[5]:


# For a data loss management (dlm) we will track data dropped every .drop() step
dlm = 0
og_len = len(df.InvoiceNo)

# It does not matter not having CustomerID in this analysis
# however a NaN Description shows us a failed transaction
# We will drop NaN CustomerID when analysing customer behavior 
df.dropna(inplace=True, subset=['Description'])

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (og_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((og_len - new_len)/og_len), (og_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows\n' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)
df.info()


# In[6]:


# Note that for dropping the rows we need the .index not a boolean list
# to_drop is a list of indices that will be used on df.drop()
to_drop = df[df.InvoiceNo.str.match('^[a-zA-Z]')].index

# Droping wrong entries starting with letters
# Our assumption is that those are devolutions or system corrections
df.drop(to_drop, axis=0, inplace=True)

# Changing data types for reducing dimension and make easier plots
df.InvoiceNo = df.InvoiceNo.astype('int64')
df.Country = df.Country.astype('category')
new_len = len(df.InvoiceNo)

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (mod_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((mod_len - new_len)/mod_len), (mod_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)


# # 3. Data visualization:

# In[7]:


# Checking categorical data from df.Country
# unique, counts = np.unique(df.Country, return_counts=True)
# print(dict(zip(unique, counts)))
country_set = df[['Country', 'InvoiceNo']]
country_set = country_set.pivot_table(columns='Country', aggfunc='count')
country_set.sort_values('InvoiceNo', axis=1, ascending=False).T


# In[8]:


# Plotting InvoiceNo distribution per Country
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
sns.countplot(y='Country', data=df);


# In[9]:


# Plotting InvoiceNo without United Kingdom
df_nUK = country_set.T.drop('United Kingdom')
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
# Note that since we transformed the index in type category the .remove_unused_categories is used
# otherwise it woul include a columns for United Kingdom with 0 values at the very end of the plot
sns.barplot(y=df_nUK.index.remove_unused_categories(), x='InvoiceNo', data=df_nUK, orient='h');


# In[10]:


# Creating subsets of df for each unique country
def df_per_country(df):
    df_dict = {}
    unique_countries, counts = np.unique(df.Country, return_counts=True)
    for country in unique_countries:
        df_dict["df_{}".format(re.sub('[\s+]', '', country))] = df[df.Country == country].copy()
        # This line is giving me the warning, I will check in further research
        # After watching Data School video about the SettingWithCopyWarning I figured out the problem
        # When doing df[df.Country == country] adding the .copy() points pandas that this is an actual copy of the original df
        df_dict["df_{}".format(re.sub('[\s+]', '', country))].drop('Country', axis=1, inplace=True)
    return df_dict

# Trick to convert dictionary key/values into variables
# This way we don't need to access dfs by df_dict['df_Australia'] for example
df_dict = df_per_country(df)
locals().update(df_dict)


# In[11]:


# Series plot function summarizing df_Countries
def series_plot(df, by1, by2, by3, period='D'):
    df_ts = df.reset_index().pivot_table(index='InvoiceDate', 
                                values=['InvoiceNo', 'Quantity', 'UnitPrice'], 
                                aggfunc=('count', 'sum'))
    df_ts = df_ts.loc[:, [('InvoiceNo', 'count'), ('Quantity', 'sum'), ('UnitPrice', 'sum')]]
    df_ts.columns = df_ts.columns.droplevel(1)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_ts.resample(period).sum().bfill()[[by1]], color='navy')
    plt.title('{}'.format(by1));
    plt.xticks(rotation=60);
    plt.subplot(2, 2, 2)
    plt.title('{}'.format(by2));
    plt.plot(df_ts.resample(period).sum().bfill()[[by2]], label='Total Sale', color='orange');
    plt.xticks(rotation=60)
    plt.tight_layout()
    
    plt.figure(figsize=(14, 8))
    plt.title('{}'.format(by3));
    plt.plot(df_ts.resample(period).sum().bfill()[[by3]], label='Total Invoices', color='green');
    plt.tight_layout()


# In[12]:


series_plot(df_UnitedKingdom, 'Quantity', 'UnitPrice', 'InvoiceNo')


# # 4. Frequent sets and association rules with apriori:

# In[13]:


# Starting preparation of df for receiving product association
# Cleaning Description field for proper aggregation 
df_UnitedKingdom.loc[:, 'Description'] = df_UnitedKingdom.Description.str.strip().copy()
# Once again, this line was generating me the SettingWithCopyWarning, solved by adding the .copy()

# Dummy conding and creation of the baskets_sets, indexed by InvoiceNo with 1 corresponding to every item presented on the basket
# Note that the quantity bought is not considered, only if the item was present or not in the basket
basket = pd.get_dummies(df_UnitedKingdom.reset_index().loc[:, ('InvoiceNo', 'Description')])
basket_sets = pd.pivot_table(basket, index='InvoiceNo', aggfunc='sum')


# In[14]:


# Apriori aplication: frequent_itemsets
# Note that min_support parameter was set to a very low value, this is the Spurious limitation, more on conclusion section
frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02) ].head()


# In[15]:


# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[['antecedants', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()


# In[17]:


# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();


# # 5. Conclusions:
# 

# ### Potential of the solution
# In Brazil we achieved impressive results in terms of applied marketing using this particular solution. Even though it's simple, one needs to take into consideration that for countries or sectors starting with analytics going from zero to an actual data-driven solution is already a game-changer.
# 
# ### Implementation simplicity
# Once again, the implementation of the solution in terms of code is simple. Deployement in most cases can be report based exporting the relevant rules for discussion.
# 
# ### Statistical interpretation
# The statistical interpretation of how support, confidence and lift can correlate with marketing strategies take some time and know how on the field where it's being applied. [Start here](http://analyticstrainings.com/?p=151) for explanations about the output attributes from this model.
# 
# ### Apriori limitations
# As seen on the KDNuggets article referenced in the Introduction, we faced the Spurious Associations limitation. This happend due to the eCommerce business model, a large number of possibilities in a single basket among an even larger number of baskets. The consequence of it is having a "sparse matrix", full of 0s which causes the support of basket occourances to drop drastically. The output achieved has its top support of 0.051 (5%).
# Such limitation might be overcome by working with the entire data set, remember that only 30000 top rows were analysed, or this could dilute the support values even more. As a last optio
