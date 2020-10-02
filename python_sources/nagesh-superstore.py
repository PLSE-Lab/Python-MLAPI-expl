#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


store["Product Name"]


# In[ ]:


y=store.sort_values("Profit", ascending=False).iloc[0:20,:]


# In[ ]:


y ["Customer Name"] # Who are the top 20 most profitable customers


# In[ ]:


store["Segment"].unique() # What is the distribution of our customer segment


# In[ ]:


z=store.sort_values(["Order Date"],ascending=True).iloc[0:20,:]


# In[ ]:


z["Customer Name"].unique() # Who are the top 20 oldest customers


# store.columns[0]

# In[ ]:


store["Customer ID"].unique()


# In[ ]:


store.sort_values(["Profit","Discount"],ascending=[True,False])


# In[ ]:


plt.xticks(rotation=90),sns.countplot("Sub-Category", data = store)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot("Category", data = store)


# In[ ]:


sns.countplot("Category",hue="Sub-Category" ,data = store)


# In[ ]:


plt.xticks(rotation=90),sns.barplot(x="Customer Name",y="Profit",data=store) # plot of top 20 most profitable customers 


# In[ ]:


sns.barplot(x="Category",y="Profit",hue="Sub-Category",data=store)


# In[ ]:


sns.countplot("Market",data=store)


# In[ ]:


sns.boxplot("Order Priority","Profit",data=store)


# In[ ]:


store.groupby("Customer ID").apply(lambda x:pd.Series(dict(store_visit=x.shape[0]))).reset_index()

