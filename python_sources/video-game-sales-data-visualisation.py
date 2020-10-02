#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ## Read the data

# In[ ]:


os.listdir("./../input")


# In[ ]:


data = pd.read_csv("../input/vgsales.csv")


# ### Analyze the data

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.tail()


# ## Group the sales data yearly and create plot

# In[ ]:


df1 = data.groupby(['Year'])


# In[ ]:


df1_mean = df1['NA_Sales','EU_Sales','JP_Sales'].aggregate(np.mean)


# In[ ]:


df1_mean.plot()
plt.title('Avg Sales')


# ### The video games sale was highest around 1990. After that it kept on decreasing.

# ## Plot frequence of the years

# In[ ]:


plt.hist(data.Year.values,bins=20)
plt.xlabel('Year')
plt.ylabel('frequency')


# ### Grouping North American & European sales data as per Genre and plotting it

# In[ ]:


df3 = data.groupby(['Genre'])


# In[ ]:


val = df3['NA_Sales'].aggregate(np.mean)


# In[ ]:


val.plot(kind='bar')
plt.xticks(rotation=30)
plt.xlabel('Genre')
plt.ylabel('NA_Sales')
plt.title('North America Sales as per Genre')


# We can observe from the above diagram that games with genre **Platform** have highest sales in North America

# ### Horizontal Plots

# In[ ]:


plt.barh(data.Genre,data.EU_Sales)
plt.xlabel('EU_Sales')
plt.ylabel('Genre')
plt.title('European Sales as per Genre')


# European sales for Sports games is highest followed by Racing and Simulation games

# In[ ]:


data.Year.max()


# # Data Summary.

# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


sns.barplot(x='Genre',y='Global_Sales',data=data)
plt.title('Global sales as per Genre')
plt.xticks(rotation=45)
plt.show()


# Platform specific games had a higher sale. However the uncertainity is also highest for same genre

# ## Scatter joint plot

# In[ ]:


sns.jointplot(x='JP_Sales',y='NA_Sales',data=data)


# Games in North America had high sales as compared to in Japan initially

# # Bar plot

# In[ ]:


sns.barplot(x='Year',y='Global_Sales',data=data)
plt.title('Global sales per year')
plt.xticks(rotation=45)
plt.show()


# Highest global sales in 1989. However the uncertainity is also highest for same genre

# # Simple Heatmap

# In[ ]:


sns.heatmap(data.corr())


# ### Group data according to top 3 publishers in North America, Euraope, Japan and Other regions

# In[ ]:


df_publishers = data.groupby('Publisher')


# In[ ]:


plot_publishers = df_publishers['NA_Sales','JP_Sales','EU_Sales','Other_Sales'].mean()


# # Boxplot

# In[ ]:


plot_publishers.boxplot()


# North American sales are highest.

# # Bar chart comparison for different regions in single figure

# In[ ]:


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

