#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


mainDf = pd.read_csv('../input/flavors_of_cacao.csv')
mainDf.columns=['company','species','REF','review_year','cocoa_p','company_location','rating','bean_typ','country']


# In[ ]:


mainDf.head(5)


# In[ ]:


choko= mainDf
d = mainDf

d.cocoa_p = d.cocoa_p.str.replace('%','')

d.cocoa_p = pd.to_numeric(d.cocoa_p)


# In[ ]:


d.info()


# In[ ]:


d.cocoa_p.hist(bins=50)


# In[ ]:


d.boxplot(column='rating')


# In[ ]:


d.describe()


# In[ ]:


d.bean_typ.value_counts()


# In[ ]:


d.country.value_counts()

