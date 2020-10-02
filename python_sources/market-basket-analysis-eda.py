#!/usr/bin/env python
# coding: utf-8

# #### Mohammed fahad vp
# ***11/01/2020***
# 
# 
# ##### Introduction
# 
# This data set was created for the purpose of learning only the ***customer segmentation*** concepts, also known as ***market basket analysis***. A wide variety of analyzes will be created in this section. However, each case will be searched and machine learning algorithms will be used.

# <img src="https://images.unsplash.com/photo-1559171667-74fe3499b5ba?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1500&q=80" width="1000px">

# In[ ]:


import numpy as np
import pandas as pd
from pandas import plotting

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[ ]:


data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


#cheking null values
data.isnull().sum().any()


# In[ ]:


#rename data's columns
data.rename(columns={'Genre':'Gender','Annual Income (k$)':'AnnualIncome','Spending Score (1-100)':'SpendingScore'},inplace=True)
data.head()


# ##### Data Exploratory Analysis

# In[ ]:


labels = ['Female', 'Male']
size = [112, 88]
colors = ['pink', 'lightblue']
explode = [0, 0.01]

plt.rcParams['figure.figsize'] = (5, 5)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('A pie chart Representing the Gender',fontsize=17)
plt.axis('off')
plt.legend()
plt.show()


# - There are more females 
# - 112 female and 88 male

# In[ ]:


data['Age'].value_counts().plot.bar(figsize = (18, 4),grid=False,legend=True,fontsize=15,color='plum')
plt.title('Distribution of age', fontsize = 17)
plt.show()


# - People at Age 32 are the Most Frequent Visitors in the Mall.
# 
# - People of Age 55, 56, 64, 69 are very less frequent in the Malls (older age,above 50s groups are lesser frequent in comparison). 
# 
# - Ages from 19 and 31 are very much frequent

# In[ ]:


data['AnnualIncome'].value_counts().plot.bar(figsize = (18, 4),grid=False,legend=True,fontsize=15,color='cyan')
plt.title('Distribution of annual income', fontsize = 17)
plt.show()


# - A wide verity of income is visible ranging from 15k to 137k
# 
# - Customers above 100k are 12
# 
# - Customers below 25k are 22
# 
# - So 166 customers have income between 25k and 100k(ie.,83%)

# In[ ]:


data['SpendingScore'].value_counts().plot.bar(figsize = (18, 4),grid=False,legend=True,fontsize=15,color='olive')
plt.title('Distribution of spending score', fontsize = 17)
plt.show()


# - There are customers having 1 spending score also, and 99 Spending score also, Which shows that the mall caters to the variety of Customers with Varying needs and requirements available in the Mall.
# 
# - Most of the Customers have their Spending Score in the range of 35-70. 

# In[ ]:


ax = sns.boxplot(data=data[['Age','AnnualIncome','SpendingScore']], orient="h", palette=sns.color_palette("hls", 5))


# In[ ]:


plt.rcParams['figure.figsize'] = (15,3)
sns.set(style = 'ticks')

plt.subplot(1,2,1)
sns.distplot(data['AnnualIncome'], rug=False, rug_kws={"color": "g"},
                  kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.title('Distribution of Annual Income', fontsize = 17)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.distplot(data['SpendingScore'], rug=False, rug_kws={"color": "r"},
                  kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.title('Distribution of spending', fontsize = 17)
plt.xlabel('Range of spending')
plt.ylabel('Count')
plt.show()


# - Here, In the above Plots we can see the Distribution pattern of Annual Income and Spending
# 
# - There are few people who earn more than 100 US Dollars. Most of the people have an earning of around 50-75 US Dollars. 
# 
# - Least Income is around 20 US Dollars. 
# 
# - The most regular customers for the Mall has age around 30-35 years of age.
# 
# - Most of the spending score is between 40 and 60

# In[ ]:


plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
sns.violinplot(x="Gender", y="SpendingScore", data=data,palette='cubehelix')
plt.title('SpendingScore & Gender',fontsize=17)


plt.subplot(1,2,2)
sns.violinplot(x="Gender", y="AnnualIncome", data=data,palette='cubehelix')
plt.title('AnnualIncome & Gender',fontsize=17)
plt.show()


# ***About spending***
# 
# - Most of the males have a Spending Score of around 25 to 70 
# 
# - Most of the females have a spending score of around 35 to 75
# 
# - Women are Shopping Leaders.
# 
# ***About annual income***
# 
# - Most of male have income 45k to 75k 
# 
# - Most of female have income 35k to 75k 

# In[ ]:



x = data['AnnualIncome']
y = data['Age']
z = data['SpendingScore']

sns.lineplot(x, y, color = 'blue')
sns.lineplot(x, z, color = 'red')
plt.title('Annual Income vs Age and Spending Score', fontsize = 20)
plt.show()


# In[ ]:


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(data=data,hue='Gender',palette='Paired',diag_kind='hist',markers='s')
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 6)
sns.heatmap(data.corr(), cmap = 'cool', annot = True,square=False)
plt.title('Heatmap for the Data', fontsize = 17)
plt.show()


# ### If you like it please Upvote.
# ### Thank you.
