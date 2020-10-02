#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# In[ ]:


path ="../input"
os.chdir(path)
sp = pd.read_csv("superstore_dataset2011-2015.csv",encoding="ISO-8859-1")


# In[ ]:


x = sp.sort_values('Profit', ascending=False)
top20 = x.head(20)
top20[['Customer Name', 'Profit']]                #top 20 profitable customers at the superstore


# In[ ]:


sns.barplot(x = "Customer Name", y= "Profit", data=top20)  # plotting of top 20 profitable customers


# In[ ]:





# In[ ]:


sns.countplot("Segment", data = sp)           #Distribution of custome Segment


# In[ ]:


sp.groupby('Customer ID').apply(lambda x: pd.Series(dict(store_visit=x.shape[0]))).reset_index()


# In[ ]:


sp1 = sp.groupby('Customer ID').apply(lambda x: pd.Series(dict(store_visit=x.shape[0]))).reset_index()
sp1.loc[sp1.store_visit == 1, ['Customer ID', 'store_visit']]     # Customers who have visited the store only once


# In[ ]:


sns.boxplot("Order Priority","Profit",data= sp)    # relationship of Order Priority and Profitability : 
                                                   # Profits slightly higher when Order priority is Medium


# In[ ]:


sns.countplot("Market",data = sp)                                # distribution of customers marketwise


# In[ ]:


sns.countplot("Region", hue= "Market", data = sp)                 # distribution of customers regionwise, marketwise


# In[ ]:


y = sp.sort_values(["Order Date"], ascending=True).iloc[0:20,:]                         # top 20 oldest customers


# In[ ]:


y["Customer Name"].unique()

