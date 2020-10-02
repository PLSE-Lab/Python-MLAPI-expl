#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import os


# In[ ]:


path="../input"


# In[ ]:


os.chdir(path)


# In[ ]:


store=pd.read_csv("superstore_dataset2011-2015.csv",encoding="ISO-8859-1")


# In[ ]:


store.info()


# In[ ]:


store.shape[0] # How many rows


# In[ ]:


store.shape[1] # How many columns


# In[ ]:


store.index.values # Get the row names


# In[ ]:


store.columns.values # Get the columns names


# In[ ]:


store.sort_values("Profit", ascending=False)


# In[ ]:


store["Product Name"]


# In[ ]:


store.columns[0]


# In[ ]:


store.sort_values(["Profit","Discount"],ascending=[True,False])


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot("Category", data = store)


# In[ ]:


sns.countplot("Category",hue="Sub-Category" ,data = store)


# In[ ]:


sns.barplot(x="Category",y="Profit",hue="Sub-Category",data=store)


# In[ ]:




