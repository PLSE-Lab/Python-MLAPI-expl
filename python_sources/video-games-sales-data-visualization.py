#!/usr/bin/env python
# coding: utf-8

# Importing Required Libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter


# Read Data File

# In[ ]:


os.listdir("../input/")
data = pd.read_csv('../input/vgsales.csv')


# Take a glance of the data

# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# #1 - Sales Region Wise Presentation In Pie Chart

# In[ ]:


plt.figure(figsize=(20, 10))
df2 = data[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum().reset_index()
df2.columns = ["Region","Total_Sales"]
plt.pie(df2.loc[:3, "Total_Sales"], labels = df2.loc[:3, "Region"])


# #2 - A Barplot Global Sales Grouped by Genre

# In[ ]:


plt.figure(figsize=(15, 10))
ax = sns.barplot("Genre", "Global_Sales",ci = None, estimator = np.sum, data = data)
ax.set_title("Global Sales By Genre")


# #3 - A Barplot to have quick glance of the region wise sales

# In[ ]:


df2 = data[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum().reset_index()
df2.columns = ["Region","Total_Sales"]
ax = sns.barplot("Region","Total_Sales", data = df2)
ax.set_title("Region Wise Global Sales")


# #4 - The year wise global sales trend.

# In[ ]:


plt.figure(figsize=(20, 10))
ax = sns.barplot("Year","Global_Sales", data= data, ci= None, estimator=np.sum)
plt.xticks(rotation=90)
ax.set_title("Year Wise Global Sales")


# #5 - Barplot Decade Wise Global Sales

# In[ ]:


data['decades'] = pd.cut(data.Year, 4, ["1980", "1990", "2000", "2020"])
ax = sns.barplot("decades", "Global_Sales", ci= None, data=data, estimator=np.sum)
ax.set_title("Decade Wise Global Sales")
plt.xticks(rotation=30)


# #6 - Determining in which decades the sales growth were high, and what kind of games people played.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot("decades","Global_Sales", hue="Genre", ci=None, estimator=np.sum, data=data, palette="muted")


# #7 - Determining in which region the the sales were high and in which years.

# In[ ]:


ax = ""
plt.figure(figsize=(20, 10))
for region in ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]:
    ax = sns.lineplot(y=region, x="Year", ci = None, data = data, legend="full", estimator=np.sum)
ax.set_title("Region Wise Sales Comparison")
ax.set_xlabel("Year")
ax.set_ylabel("Sales in Millions")
ax.legend(["NA_Sales","EU_Sales","JP_Sales","Other_Sales"])


# #8 - Which platforms people liked the most.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot("Platform","Global_Sales", ci=None, estimator=np.sum, data=data, palette="muted")


# #9 - Category of games made the highest sales in the years.

# In[ ]:


plt.figure(figsize=(20, 10))
df = data.groupby(["Year","Genre","Platform"])
sns.barplot("Year", "Global_Sales", hue="Genre", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(50).reset_index(),  palette="muted")


# #10 - The highest markest share of the publishers in global sales.

# In[ ]:


plt.figure(figsize=(20, 10))
df = data.groupby(["Publisher"])
sns.barplot("Global_Sales", "Publisher", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(20).reset_index(),  palette="muted")


# #11 - The games people enjoyed the most.

# In[ ]:


plt.figure(figsize=(20, 10))
df = data.groupby(["Name"])
sns.barplot("Global_Sales", "Name", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(20).reset_index(),  palette="muted")

