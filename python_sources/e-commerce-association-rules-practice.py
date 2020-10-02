#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


PATH_INPUT = "/kaggle/input/"
PATH_WORKING = "/kaggle/working/"
PATH_TMP = "/tmp/"


# ## Reading data

# In[ ]:


df_raw = pd.read_csv(f'{PATH_INPUT}data.csv', encoding='iso-8859-1')


# In[ ]:


df_raw.shape


# In[ ]:


get_ipython().system('ls -lh {PATH_INPUT}')


# In[ ]:


df_raw.describe(include='all')


# We have
# * 25900 invoices
# * 4070 stock codes
# * 4223 description

# # Pre-processing 

# ## Data cleaning

# In[ ]:


df = df_raw.copy()


# *Ideas: merge the cancelling order back to the original order (not implemented)*

# In[ ]:


df.query('Quantity < -80000')


# In[ ]:


df.query('Quantity > 80000')


# ## Data transformation

# In[ ]:


basket = df.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack()
basket.shape


# Shape is alright

# Replace with 0 and 1

# In[ ]:


basket = basket.applymap(lambda x: 1 if x>0 else 0)


# In[ ]:


basket.head(1)


# In[ ]:


basket.iloc[0].value_counts()


# Success

# # Modeling

# In[ ]:


get_ipython().run_cell_magic('time', '', 'itemsets = apriori(basket, min_support=0.005, use_colnames=True)')


# In[ ]:


itemsets.shape


# In[ ]:


itemsets.sort_values('support',ascending=False).head()


# In[ ]:


rules = association_rules(itemsets, metric="lift", min_threshold=1)


# In[ ]:


rules.shape


# In[ ]:


sns.scatterplot(x='support', y='confidence', hue='lift', data=rules)
plt.show()


# In[ ]:


sns.scatterplot(x='support', y='confidence', hue='leverage', data=rules)
plt.show()


# In[ ]:


rules.sort_values('lift', ascending=False).head(10)


# In[ ]:




