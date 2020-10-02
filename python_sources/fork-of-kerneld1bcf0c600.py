#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df=pd.read_csv("../input/groceries_200.csv",header=-1)

# Any results you write to the current directory are saved as output.


# In[ ]:


df.head()


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


arr=np.array(df.values)


# In[ ]:


all_items=np.unique(arr[~(arr==0)])
all_items


# In[ ]:


basket=pd.DataFrame(0,index=np.arange(len(df)),columns=all_items)


# In[ ]:


basket.head()


# In[ ]:


for col in df.columns:       # columnwise
    t_f = (df[col].values!=0)# 
    itms = df.iloc[t_f,col]# all the elements which are not zero in col 
    #|| first element of every basket
    indx = itms.index         # save the index of above elements  
    for i in range(len(itms)):
        basket.loc[indx[i]][itms.iloc[i]] += 1


# In[ ]:


basket.head()


# In[ ]:


frequent_itemsets = apriori(basket,min_support=0.02,use_colnames=True)
frequent_itemsets.head()


# In[ ]:


frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets.tail()


# In[ ]:


rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1)

rules.head()


# In[ ]:


len(rules)


# In[ ]:


rules = association_rules(frequent_itemsets,metric="lift", min_threshold=5)
len(rules)
rules


# In[ ]:


rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1)
# where confidence is greater than 0.7
rules[rules['confidence']>=0.7]


# In[ ]:



rules = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.5)
rules.head()


# In[ ]:


rules = association_rules(frequent_itemsets, min_threshold=0.5)
rules.head()


# In[ ]:


rules = association_rules(frequent_itemsets, min_threshold=2.0)
rules.head()


# In[ ]:


rules = association_rules(frequent_itemsets,metric="support", min_threshold=0.01)
rules.head()


# In[ ]:


rules = association_rules(frequent_itemsets,metric="support", min_threshold=0.02)
rules.head()


# In[ ]:




