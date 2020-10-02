#!/usr/bin/env python
# coding: utf-8

# #__Importing necessary libraries and packages__

# In[84]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# #**Load in the dataset**

# In[85]:


os.chdir("../input")


# In[86]:


data =pd.read_csv("vgsales.csv")


# #**Get to know about the data**

# In[87]:


data.shape


# In[88]:


data.head()


# In[89]:


data.columns


# In[90]:


data.tail()


# #**Exploring data**

# In[91]:


data.describe()


# In[92]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,cmap='RdBu_r')


# ###**Positive values shows positive correlation whereas negative values shows inverse correlation. As we can see North American sales has contributed considerably to overall Global sales;hence we can see a positive correlation between the two.**

# #**Grouping sales data of each region yearly and plotting the same**

# In[93]:


df1 = data.groupby(['Year'])
plt.figure(figsize=(10,10))
df1_mean = df1['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)
df1_mean.plot(figsize=(10,10))
plt.title('Average sales over the course of years')


# ###**As we can see the video games sale was at its peak around 1990 which went on decreasing over the course of coming years**

# #**Plotting frequency of the years in dataset**

# In[94]:


plt.figure(figsize=(10,10))
plt.hist(data.Year.values,bins=20)
plt.xlabel('Year')
plt.ylabel('frequency')


# #**Grouping North American, Japan, European & Other region sales data as per Genre and plotting it**

# In[95]:


data['Genre'].unique()


# In[96]:


df3 = data.groupby(['Genre'])
val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)
val.plot(kind='bar',figsize=(20,8))
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Sale of games in each region',fontsize=16)
plt.title('Sales as per Genre',fontsize=16)


# ###**Platform specific games have highest sales in North America followed by Shooter and Sports games. In European regions Shooter games have slightly higher demand than Platform specific ones. Japan has a higher Role-playing games.**

# #**Grouping North American, Japan, European & Other region sales data as per Platform and plotting it**

# In[97]:


data['Platform'].unique()


# In[98]:


df3 = data.groupby(['Platform'])

val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)
val.plot(kind='bar',figsize=(20,8))
plt.xlabel('Platform',fontsize=16)
plt.ylabel('Sale of games in each region',fontsize=16)
plt.title('Sales as per Platform',fontsize=16)


# ###**NES shows highest sales in all regions followed by GB. 2600 shows noticeable sale in North American region as compared to other regions.**

# #**Plotting Sales in all regions as per Platform**

# In[99]:


df3 = data.groupby(['Platform'])
val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)
plt.figure(figsize=(12,10))
ax = sns.boxplot(data=val, orient='h')
plt.xlabel('Revenue per game',fontsize=16)
plt.ylabel('Region',fontsize=16)
plt.title('Distribution of sales as per Platform',fontsize=16)


# In[100]:


data.Year.max()


# #**Plotting Genres included in the dataset and controlling graph using matplotlib functions**

# In[101]:


plt.figure(figsize=(12,8))
sns.countplot(x='Genre',data=data)
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Count',fontsize=16)
plt.show()


# ###**Games belonging to Action genre are sold in highest numbers**

# In[102]:


plt.figure(figsize=(12,8))
sns.barplot(x='Genre',y='Global_Sales',data=data)
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Global Sales',fontsize=16)
plt.title('Global sales as per Genre',fontsize=16)
plt.show()


# ###**Platform specific games had a higher sale. However the uncertainity is also highest for same genre**

# #**Scatter joint plot**

# In[103]:


sns.jointplot(x='NA_Sales',y='Global_Sales',data=data)


# ###**Games in North America had high sales which in turn contributed to the overall Global sales.**

# #**Bar plot**

# In[104]:


plt.figure(figsize=(20,8))
sns.barplot(x='Year',y='Global_Sales',data=data)
plt.title('Global sales per year')
plt.xticks(rotation=45)
plt.show()


# ###**Highest global sales in 1989. However the uncertainity is also highest for same genre**

# #**Group data according to top 3 publishers in North America, Europe, Japan and Other regions**

# In[105]:


df_publishers = data.groupby('Publisher')
plot_publishers = df_publishers['NA_Sales','JP_Sales','EU_Sales','Other_Sales'].mean()
plt.figure(figsize=(12,8))
plot_publishers.boxplot()


# ###**North American sales are highest; although outerlayers are high as well**

# #**Bar chart comparison for different regions in single figure**

# In[106]:


sort_publishers = plot_publishers.sort_values('EU_Sales',ascending=False)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,4,1)
ax1.set_xticklabels(labels = 'European Union', rotation=90)
sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).EU_Sales)
plt.title('European Union')
plt.ylabel('Revenue')
plt.suptitle('Revenues per region',size=22)
sort_publishers = plot_publishers.sort_values('NA_Sales',ascending=False)
ax2 = fig.add_subplot(1,4,2,sharey=ax1)
ax2.set_xticklabels(labels = 'North America', rotation=90)
sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).NA_Sales)
plt.title('North America')
plt.ylabel('Revenue')
sort_publishers = plot_publishers.sort_values('JP_Sales',ascending=False)
ax3 = fig.add_subplot(1,4,3,sharey=ax1)
ax3.set_xticklabels(labels = 'Japan', rotation=90)
sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).JP_Sales)
plt.title('Japan')
plt.ylabel('Revenue')
sort_publishers = plot_publishers.sort_values('Other_Sales',ascending=False)
ax4 = fig.add_subplot(1,4,4,sharey=ax1)
ax4.set_xticklabels(labels = 'Japan', rotation=90)
sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).Other_Sales)
plt.title('Other Regions')
plt.ylabel('Revenue')

