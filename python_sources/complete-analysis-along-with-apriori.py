#!/usr/bin/env python
# coding: utf-8

# ## **Transactions from a bakery**
# 
# ### **Context**
# #### The data belongs to a bakery called "The Bread Basket", located in the historic center of Edinburgh. This bakery presents a refreshing offer of Argentine and Spanish products.
# 
# ****

# In[ ]:


# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Reading the data
data = pd.read_csv("../input/BreadBasket_DMS.csv")


# In[ ]:


# Listing the data columns
data.info()


# **File contains 21293 records**

# In[ ]:


# Describing the quantitative column 
data.describe()


# **Not much variance in Mean and Median** 
# 
# >  No initial outlier detected

# In[ ]:


# look at head
data.head()


# ## Missing Value Treatment
# 
# Let's now move to missing value treatment. 
# 
# Let's have a look at the number of missing values.

# In[ ]:


data.isnull().sum()


# No Missing value detected

# ### Convert data & time to datatime datatype

# In[ ]:


data['Date time']= pd.to_datetime(data['Date']+' '+data['Time'])


# ### Data range check

# In[ ]:


data['Date time'].dt.year.value_counts()


# In[ ]:


data['Year Month']=data['Date time'].map(lambda x: 100*x.year + x.month)
data['Hour']=data['Date time'].dt.hour
data['Day']=data['Date time'].dt.weekday_name
data['Weekend vs Weekday'] = data['Date time'].apply(lambda x: 'Weekend' if x.dayofweek//5==1 else 'Weekday')


# ## Univariate Analysis

# In[ ]:


plt.figure(figsize=[10,5])
plt.plot(data['Date time'], data['Transaction'])
plt.title('No of Transaction by DateTime')


# In[ ]:


Transaction_by_month=data[['Year Month','Transaction']].groupby('Year Month',as_index=False).sum()


# In[ ]:


plt.figure(figsize=[10,5])
sns.barplot(x='Year Month',y='Transaction',data=Transaction_by_month)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by month')


# Data is available from 30/10/2016 to 09/04/2017.

# In[ ]:


plt.figure(figsize=[10,5])
sns.boxplot(x='Day',y='Transaction',data=data)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Day')


# In[ ]:


plt.figure(figsize=[10,5])
sns.boxplot(x='Weekend vs Weekday',y='Transaction',data=data)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Weekend vs Weekday')


# In[ ]:


plt.figure(figsize=[10,5])
plt.ticklabel_format(style='plain', axis='y')
plt.title('Sale by Hour')
plt.plot(data[['Hour','Transaction']].groupby('Hour').sum())


# We can notice most sales happens between 6 AM to 6 PM

# **Histogram**

# In[ ]:


plt.figure(figsize=[10,5])
sns.distplot(data['Transaction'],bins=100)


# No insights found from above histogram

# In[ ]:


Item_by_transaction=data[['Item','Transaction']].groupby('Item',as_index=False).sum().sort_values(by='Transaction',ascending=False)


# In[ ]:


Item_by_transaction['Transaction %']=Item_by_transaction['Transaction']/Item_by_transaction['Transaction'].sum()


# In[ ]:


plt.figure(figsize=[10,5])
sns.barplot(x='Item',y='Transaction',data=Item_by_transaction.head(10))
plt.ticklabel_format(style='plain', axis='y')
plt.title('Top 10 Items')
plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize=[10,5])
sns.barplot(x='Item',y='Transaction %',data=Item_by_transaction.head(10))
plt.ticklabel_format(style='plain', axis='y')
plt.title('Top 10 Items')
plt.xticks(rotation = 90)


# In[ ]:


Hour_by_Item=data[['Hour','Item','Transaction']].groupby(['Hour','Item'],as_index=False).sum()


# In[ ]:


Top_items=list(Item_by_transaction['Item'].head(10))


# In[ ]:


plt.figure(figsize=[10,5])
sns.boxplot(x='Item',y='Transaction',data=data[data['Item'].isin(Top_items)])
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Top 10 Item')


# ## Bivariate Analysis

# In[ ]:


plt.figure(figsize=[13,5])
plt.ticklabel_format(style='plain', axis='y')
plt.title('Sale by Hour for Top 5 Items')
sns.lineplot(x='Hour',y='Transaction',data=Hour_by_Item[Hour_by_Item['Item'].isin(Top_items)],hue='Item')


# In[ ]:


Top25Items=list(Item_by_transaction['Item'].head(25))
dataTop25Items=data[data['Item'].isin(Top25Items)]
dataTop25Items_pivot = dataTop25Items[['Date','Item','Transaction']].pivot_table('Transaction', 'Date', 'Item')


# ### Correlation Plot for Top 25 Items

# In[ ]:


# Correlation Plot
f, ax = plt.subplots(figsize=[12,10])
sns.heatmap(dataTop25Items_pivot.corr(),annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot")
plt.show()


# ## **** Conclusion based on Correlation****
# * ### Coffee & Bread is correlated with all Items, Hence bread always bought with all other items
# * ### Baguette not bought often with other Items 
# * ### Tiffin &  Baguette not bought together often

# ## Using Apriori Algorithm

# In[ ]:


dataTop25Items_unstack=dataTop25Items.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
dataTop25Items_unstack = dataTop25Items_unstack.applymap(lambda x: 0 if x<=0 else 1)


# In[ ]:


dataTop25Items_frequent = apriori(dataTop25Items_unstack, min_support=0.01, use_colnames=True)


# #### Confidence > 40%

# In[ ]:


assciation = association_rules(dataTop25Items_frequent, metric="lift", min_threshold=1)
assciation[assciation['confidence']>=0.4]


# ## **** Conclusion based on Apriori****
# * ### Coffee is common item bought with other items
# * ### Alfajores & Coffee is the most common combo
# * ### Toast &  Coffee is the least most common combo

# In[ ]:




