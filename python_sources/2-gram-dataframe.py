#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


file="../input/wp_2gram.txt"


# In[ ]:


data=pd.read_csv(file,header=None,sep='\t')


# In[ ]:


data.head()


# In[ ]:


data[1]=data[1]+" "+data[2]


# In[ ]:


data.head() 


# In[ ]:


data.drop(2,axis=1)


# In[ ]:


data.to_csv("2_gram.csv")


# In[ ]:




