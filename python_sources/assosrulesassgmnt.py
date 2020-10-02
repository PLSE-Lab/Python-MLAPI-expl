#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# In[ ]:


get_ipython().system('pip install mlxtend')


# In[ ]:


df = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv',names=['products'],header=None)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


df.values


# In[ ]:


data = list(df["products"].apply(lambda x:x.split(',')))
data 


# In[ ]:


from mlxtend.preprocessing import TransactionEncoder


# In[ ]:


te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df


# In[ ]:


from mlxtend.frequent_patterns import apriori


# In[ ]:


df1 = apriori(df,min_support=0.01,use_colnames=True)
df1


# In[ ]:


df1.sort_values(by="support",ascending=False)


# In[ ]:


df1['length'] = df1['itemsets'].apply(lambda x:len(x))
df1


# In[ ]:


df1[(df1['length']==2) & (df1['support']>=0.15)]


# In[ ]:


k = association_rules(df1, metric = "confidence", min_threshold = 0.15)
print(k.head(10))


# In[ ]:



# CORNFLAKES and COFFEE are observed together in 20 percent of all purchases
# 50 percent of customers who buy #CORNFLAKES also get biscuits.
# CORNFLAKES increase sales of the biscuit products by 1.42 times in purchases


# In[ ]:


#action idea:
#CORNFLAKES and BISCUIT can be packed and labeled together, stacked on the shelves and presented to the customer.

