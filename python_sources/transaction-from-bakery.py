#!/usr/bin/env python
# coding: utf-8

# # Transaction from the bakery

# ## Data is about transactions happened in a bakery for each items with a timestamp  It has 21,294 observations, over 9000 transactions and 94 unique items.
# 

# ## Import the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#we need to install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ## Import the dataset
# 
# 

# In[ ]:


df=pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


print(df.info())


# ## Data Engineering
# checking for null values
# print(df.isnull().sum().sort_values(ascending=False))

# In[ ]:


df.head(10)


# ### Adding year , month and day
# In the above, we have date format column that need to dissove into year ,month and day.

# In[ ]:


df['Year']=df['Date'].apply(lambda x:x.split("-")[0]) #year
df['Month']=df['Date'].apply(lambda x:x.split("-")[1]) #month
df['Day']=df['Date'].apply(lambda x:x.split("-")[2]) #day
df['Hour']=df['Time'].apply(lambda x:x.split(":")[0]) #Hour
df['Minute']=df['Time'].apply(lambda x:x.split(":")[1]) #minutes
df['Seconds']=df['Time'].apply(lambda x:x.split(":")[2]) #seconds


# In[ ]:


df.head(10)


# ## Visualizing and Understanding the data

# In[ ]:


sold = df['Item'].value_counts()
sold.head(10)


# In[ ]:


#visualization
import seaborn as sns
sns.set(style ='whitegrid')


# ### Visulaization for month wise and year wise comparison over transactions

# In[ ]:


ax = sns.barplot(x='Month', y='Transaction', data=df)


# #### As per the data, 01-04 represents jan2018 to apr2018 and 10-12 represents 30 Oct 2017 - Dec 2017. Sales in the bakery getting higher and linearly increasing with 20% 

# #### In day wise chart, we can see that distribution of transaction over days of month

# In[ ]:


ax = sns.barplot(x='Year', y='Transaction', data=df)


# #### Hour wise chart represents 24 hour time period and sometimes shop opens upto 1AM. From 8AM -5PM, there is more or less same trend of transactions happen in the bakery. But after 6PM, transactions get doubled when compare with afternoon time transactions. 

# In[ ]:




ax = sns.barplot(x='Hour', y='Transaction', data=df)


# #### Most of items sold as per the transactions happenned. Below chart has top 20 items sold where has 

# In[ ]:


items_sold = sold.head(20)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
items_sold.plot(kind='bar')
plt.title('Sold Items')


# ## Market Basket Analysis

# In[ ]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori


# In[ ]:


transaction_list = []

# For loop to create a list of the unique transactions throughout the dataset:
for i in df['Transaction'].unique():
    tlist = list(set(df[df['Transaction']==i]['Item']))
    if len(tlist)>0:
        transaction_list.append(tlist)
print(len(transaction_list))


# In[ ]:


te = TransactionEncoder()
te_ary=te.fit(transaction_list).transform(transaction_list)
df2=pd.DataFrame(te_ary, columns = te.columns_)


# In[ ]:


# Apply apriori algorithm to know how items are bought together by fixing more than 1 as limit and sort the values by confidence 


# In[ ]:


items = apriori(df2, min_support=0.03, use_colnames=True)
rules = association_rules(items, metric='lift', min_threshold=1.0)
rules.sort_values('confidence', ascending=False)


# ## Conclusion
# 
# Finally , I have finished the analysis and this is my first kernel. I have took inputs from other kernels and analyze ths data. From the above output, we can see that "Coffee" , "Pastry" are making more transaction with other items. So this will help the business like they can place higher confidence level items near to each other so that it will boost the sales of the bakery. 
# 
# As I shown in the visualization for dataframe, it will improve the understanding of business over time period and can make better decisions with respect to trend. If any changes need to make in this Notebook please comment and upvote . 
