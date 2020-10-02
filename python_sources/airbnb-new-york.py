#!/usr/bin/env python
# coding: utf-8

# # Airbnb New York (Manhattan-Brooklyn)

# ## In this kernel, I worked on Airbnb (New York) open data. I visualized the price distributions of Manhattan and Brooklyn. You can find the dataset at the bottom of the page.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns
sns.set()


# ## Import Data

# In[ ]:


raw_data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
raw_data.head()


# ## Dealing with missing values

# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data.dropna(inplace=True)


# In[ ]:


data = raw_data.copy()


# In[ ]:


data.reset_index(drop=True, inplace=True)


# ## Explore the PDFs

# In[ ]:


sns.distplot(data["price"])


# ## Dealing with Outliers

# In[ ]:


q = data["price"].quantile(0.98)
data_1 = data[data["price"]<q]
sns.distplot(data_1["price"])


# In[ ]:


data_1.reset_index(drop=True, inplace=True)


# ## 5 Most Expensive Neighbourhoods in Manhattan

# In[ ]:


midtown = data_1[data_1["neighbourhood"]=="Midtown"]["price"].sum() ## Midtown Total Price
chinatown = data_1[data_1["neighbourhood"]=="Chinatown"]["price"].sum() ## Chinatown Total Price
roosevelt = data_1[data_1["neighbourhood"]=="Roosevelt Island"]["price"].sum() ## Roosevelt Island Total Price
murray = data_1[data_1["neighbourhood"]=="Murray Hill"]["price"].sum() ## Murray Hill Total Price
manPrice = midtown + chinatown + roosevelt + murray ## Manhattan Total Price


# ## 5 Most Expensive Neighbourhoods in Brooklyn

# In[ ]:


clinton = data_1[data_1["neighbourhood"]=="Clinton Hill"]["price"].sum() ## Clinton Hill Total Price
columbia = data_1[data_1["neighbourhood"]=="Columbia St"]["price"].sum() ## Columbia St. Total Price
kensington = data_1[data_1["neighbourhood"]=="Kensington"]["price"].sum() ## Kensington Total Price
williamsburg = data_1[data_1["neighbourhood"]=="Williamsburg"]["price"].sum() ## Williamsburg Total Price
brkPrice = clinton + columbia + kensington + williamsburg ## Brooklyn Total Price


# ## Pie Plot of Each Neighbourhood Group

# In[ ]:


percMan = 100 / manPrice ## Percentage for Manhattan
percBrk = 100 / brkPrice ## Percentage for Brooklyn

sizes1 = [midtown*percMan, chinatown*percMan, roosevelt*percMan, murray*percMan]
sizes2 = [clinton*percBrk, columbia*percBrk, kensington*percBrk, williamsburg*percBrk]

f, (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(18,20))
ax1.pie(sizes1, labels=("Midtown", "Chinatown", "Roosevelt Island", "Murray Hill"), colors=["yellow", "green","gray","red"], autopct="%1.1f%%")
ax1.set_title("Price Distribution of Manhattan",size=18)
ax2.pie(sizes2, labels=("Clinton Hill", "Columbia St", "Kensington","Williamsburg"), colors=["yellow", "green","gray","red"], autopct="%1.1f%%")
ax2.set_title("Price Distribution of Brooklyn",size=18)
plt.show()

