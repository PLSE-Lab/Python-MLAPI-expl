#!/usr/bin/env python
# coding: utf-8

# ## Importing the modules needed

# In[ ]:


import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.__version__


# ## Importing the data

# In[ ]:



import os
print(os.listdir("../input"))
#We can see that the dataset has data of Data,Time,Transaction and the item sold at the bakery.
df = pd.read_csv('../input/BreadBasket_DMS.csv')

df.head(), df.info()


# ## Transforming all item names to lower case

# In[ ]:


df['Item'] = df['Item'].str.lower()


# ## Inspecting the data

# In[ ]:


x = df['Item'] == "none"
print(x.value_counts())


# This means that there rows where transaction is made but item is "none" and number of such rows are 786. which will be removed to take in consideration only those rows where transaction is made with an item.

# ## Droping all none values

# In[ ]:


df = df.drop(df[df.Item == 'none'].index)


# ## Checking all unique items that are sold

# In[ ]:


len(df['Item'].unique())


# ### There are 94 different unique items sold by bakery or simply only these items are present in the Items column.

# ## Top 20 best selling items

# In[ ]:


fig, ax=plt.subplots(figsize=(16,7))
df['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=17)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=20)
plt.grid()
plt.ioff()


# Using Datetime i created a new column called "day_of_week" which can give us insights on which weekday has more transactions

# In[ ]:


df['datetime'] = pd.to_datetime(df['Date']+" "+df['Time'])
df['Week'] = df['datetime'].dt.week
df['Month'] = df['datetime'].dt.month
df['Weekday'] = df['datetime'].dt.weekday
df['Hours'] = df['datetime'].dt.hour


# In[ ]:


df1=df[['Date','Transaction', 'Month','Week', 'Weekday','Hours']]


# In[ ]:


df2['Counts'] = df1(['Date']).size().reset_index(name="counts")


# In[ ]:





# In[ ]:


sns.countplot(x='Weekday',data=df1)


# In[ ]:





# In[ ]:


sns.countplot(x='Hours',data=df1)


# Now, we need to run apriori algorithm to get insight that if a customer buys one item which item he/she buys next.

# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


hot_encoded_df = df.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')


# In[ ]:


hot_encoded_df.head()


# Above lineAbove line of code is transfrom data to make items as columns and each transaction as a row and count same Items bought in one transaction but fill other cloumns of the row with 0 to represent item which are not bought.
# 

# In[ ]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
hot_encoded_df = hot_encoded_df.applymap(encode_units)


# In[ ]:


frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)


# Support is an indication of how frequently the itemset appears in the dataset.
# 
# Confidence is an indication of how often the rule has been found to be true.

# In[ ]:


rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules.head(10)


# We only want to see the rules where confidence is greater than or equal to 50% so:

# In[ ]:


rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5)]


# For instance from the last rule we can see that toast and coffee are commonly bought together. This makes sense since people who purchase toast would like to have coffee with it. 
# 
# The support value for the this rule is 0.023666. This number is calculated by dividing the number of transactions containing toast divided by total number of transactions. The confidence level for the rule is 0.704403 which shows that out of all the transactions that contain toast , 70.44% of the transactions also contain coffee. Finally, the lift of 1.47 tells us that coffee is 1.47 times more likely to be bought by the customers who buy toast compared to the default likelihood of the sale of coffee.

# In[ ]:


support = rules.as_matrix(columns=['support'])
confidence = rules.as_matrix(columns=['confidence'])
import seaborn as sns

for i in range (len(support)):
    support[i] = support[i]
    confidence[i] = confidence[i]
    
plt.title('Assonciation Rules')
plt.xlabel('support')
plt.ylabel('confidance')
sns.regplot(x=support, y=confidence, fit_reg=False)

