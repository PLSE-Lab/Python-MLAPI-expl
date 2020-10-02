#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/breadbasketanalysis/BreadBasket_DMS.csv")


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df['Item'].unique()


# In[ ]:


df.drop(df[df['Item']=='NONE'].index, inplace=True)


# In[ ]:


# Year
df['Year'] = df['Date'].apply(lambda x: x.split("-")[0])
# Month
df['Month'] = df['Date'].apply(lambda x: x.split("-")[1])
# Day
df['Day'] = df['Date'].apply(lambda x: x.split("-")[2])


# In[ ]:


#top 15 most sold items
ms = df['Item'].value_counts().head(15)
print(ms)


# In[ ]:


#Monthly transactions
df.groupby('Month')['Transaction'].nunique().plot(kind='bar', title='Monthly Sales')
plt.show()


# In[ ]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori


# In[ ]:


unq_transactions = []

for i in df['Transaction'].unique():
    tlist = list(set(df[df['Transaction']==i]['Item']))
    if len(tlist)>0:
        unq_transactions.append(tlist)
print(len(unq_transactions))


# In[ ]:


te = TransactionEncoder()
te_ary = te.fit(unq_transactions).transform(unq_transactions)
df2 = pd.DataFrame(te_ary, columns=te.columns_)


# In[ ]:


df2.head()


# In[ ]:


frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
rules.sort_values('confidence', ascending=False)


# In[ ]:




