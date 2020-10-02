#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


df['Item']=df['Item'].str.lower()


# In[ ]:


x=df['Item']== 'none'
print(x.value_counts())


# This means that there rows where transaction is made but item is "none" and number of such rows are 786. which will be removed to take in consideration only those rows where transaction is made with an item.

# In[ ]:


df=df.drop(df[df.Item == 'none'].index)


# In[ ]:


len(df['Item'].unique())


# There are 94 different unique items sold by bakery or simply only these items are present in the Items column.

# In[ ]:


df_for_top10_Items=df['Item'].value_counts().head(10)
Item_array= np.arange(len(df_for_top10_Items))


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
Items_name=['coffee','bread','tea','cake','pastry','sandwich','medialuna','hot chocolate','cookies','brownie']
plt.bar(Item_array,df_for_top10_Items.iloc[:])
plt.xticks(Item_array,Items_name)
plt.title('Top 5 most selling items')
plt.show()


# Wow! Didn't thought the amount of coffee sold is this much over other stuff
# 
# ![Coffee](https://media.giphy.com/media/h5LHSr2Sgd4xq/giphy.gif)

# Using Datetime i created a new column called "day_of_week" which can give us insights on which weekday has more transactions

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'],format= '%H:%M:%S' ).dt.hour
df['day_of_week'] = df['Date'].dt.weekday
d=df.loc[:,'Date']


# In[ ]:


weekday_names=[ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
Weekday_number=[0,1,2,3,4,5,6]
week_df = d.groupby(d.dt.weekday).count().reindex(Weekday_number)
Item_array_week= np.arange(len(week_df))


# In[ ]:


plt.figure(figsize=(15,5))
my_colors = 'rk'
plt.bar(Item_array_week,week_df, color=my_colors)
plt.xticks(Item_array_week,weekday_names)
plt.title('Number of Transactions made based on Weekdays')
plt.show()


# This graph gives us clear insight that people shop/make transactions more towards the weekends.Now lets see in which hours of a day people make more transactions.

# In[ ]:


dt=df.loc[:,'Time']
Hour_names=[ 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
time_df=dt.groupby(dt).count().reindex(Hour_names)
Item_array_hour= np.arange(len(time_df))


# In[ ]:


plt.figure(figsize=(15,5))
my_colors = 'rb'
plt.bar(Item_array_hour,time_df, color=my_colors)
plt.xticks(Item_array_hour,Hour_names)
plt.title('Number of Transactions made based on Hours')
plt.show()


# Now, we need to run apriori algorithm to get insight that if a customer buys one item which item he/she buys next .

# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


hot_encoded_df=df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')


# Above line of code is transfrom data to make items as columns and each transaction as a row and count same Items bought in one transaction but fill other cloumns of the row with 0 to represent item which are not bought.

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


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(10)


# We only want to see the rules where confidence is greater than or equal to 50% so:

# In[ ]:


rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]


# For instance from the last rule we can see that toast and coffee are commonly bought together. This makes sense since people who purchase toast would like to have coffee with it. i.e some people buy baggle/toast/cookie/scone and coffee togather
# 
# The support value for the this rule is 0.023666. This number is calculated by dividing the number of transactions containing toast divided by total number of transactions. The confidence level for the rule is 0.704403 which shows that out of all the transactions that contain toast , 70.44% of the transactions also contain coffee. Finally, the lift of 1.47 tells us that coffee is 1.47 times more likely to be bought by the customers who buy toast compared to the default likelihood of the sale of coffee.

# In[ ]:


support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])
import seaborn as sns
 
for i in range (len(support)):
    support[i] = support[i] 
    confidence[i] = confidence[i] 
     
plt.title('Association Rules')
plt.xlabel('support')
plt.ylabel('confidence')    
sns.regplot(x=support, y=confidence, fit_reg=False)
 

