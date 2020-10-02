#!/usr/bin/env python
# coding: utf-8

# # Analysing Windows Store
# 
# This is an EDA on the Data of apps in the **Microsoft Windows Store** reviews are welcome .
# 
# ![Microsoft Store](https://cdn.pixabay.com/photo/2013/02/12/09/07/microsoft-80658_1280.png)                     
# 
# 
# 

# # Importing the necessary libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
size=15
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25,
          'font.size':18
         
         }
plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')
paths = []


# # Reading csv

# In[ ]:


df = pd.read_csv('/kaggle/input/windows-store/msft.csv')
df.dropna(inplace=True)


# ## Seeing the bare Data

# In[ ]:


df.head(10)


# ## Seeing the distribution of Free apps and Paid Apps

# In[ ]:


index = df.index
num_of_rows = len(index)
num_of_rows


# In[ ]:


free_apps=(df["Price"] == 'Free').sum() / num_of_rows * 100
paid_apps = (df["Price"] != "Free").sum() / num_of_rows*100


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))


data = [free_apps,paid_apps]
labels = ["Free Apps","Paid Apps"]

def func(pct):

    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct),
                                  textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Distribution of Apps Price",
          loc="center",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=20, weight="bold")

ax.set_title("Distribution of Apps Price")

plt.show()


# # Inference
# 
#  - 97 % of the apps are free apps

# In[ ]:


categories = df["Category"].unique()


# In[ ]:


categories


# In[ ]:


categories_dict = dict()
for item in categories:
    categories_dict[item]= (df['Category'] == item).sum()


# In[ ]:


categories_dict


# In[ ]:


values = list(categories_dict.values())
labels = list(categories_dict.keys())
values


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

def func(pct):

    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct),
                                  textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Distribution of Categories",
          loc="center",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=20, weight="bold")

ax.set_title("Distribution of  Categories")
plt.show()


# # Inference
# 
#   - Music Category has the nost. number of apps

# In[ ]:


years = [ '2011','2014','2015','2016','2017','2018','2019']


# In[ ]:


years_dict =dict()
for year in years:
    years_dict[year] = (df['Date'].str.contains(year)).sum()
    


# In[ ]:


years_dict


# In[ ]:


plt.pie(years_dict.values(),labels=years_dict.keys(),autopct=lambda pct: func(pct))
plt.title("Year wise Distribution")

