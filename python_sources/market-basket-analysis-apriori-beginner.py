#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


data = data[data['Item']!='NONE']


# In[ ]:


count = data["Item"].value_counts()[:15]


# In[ ]:


sns.barplot(x = count.index,y=count.values,alpha=0.6)
sns.set(rc={'figure.figsize':(27.7,18.27)})


# In[ ]:


data['cnt']=1


# In[ ]:


table = data.groupby(['Transaction','Item'])['cnt'].sum().unstack()


# In[ ]:


def boolean(x):
    if x>0:
        return 1 
    else:
        return 0
table = table.applymap(boolean)    


# In[ ]:


bb= apriori(table,min_support=0.001,use_colnames=True)


# In[ ]:


rules_store = association_rules(bb,min_threshold=0.6)


# In[ ]:


rules_store


# In[ ]:




