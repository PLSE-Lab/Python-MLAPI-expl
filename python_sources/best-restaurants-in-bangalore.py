#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from collections import Counter
from nltk import word_tokenize, sent_tokenize
import seaborn as sns
print(os.listdir("../input"))


# ### Import data

# In[ ]:


data=pd.read_csv("../input/zomato.csv")


# In[ ]:


data.head()


# In[ ]:


print(len(data))


# In[ ]:


data.info()


# ### Data analysis

# ### 1. What are the top cuisines are in Bangalore?

# In[ ]:


data['cuisines']=[str(x) for x in data['cuisines']] # string the float item in data["cuisines"]
cuisines=",".join(data["cuisines"]) # include all the cuisines in a big string
cuisines= [x.strip() for x in cuisines.split(',')] # get rid of the whitespace and seperate all the item by comma

cuisines=Counter(cuisines)
top_10_cuisines=[x for x in cuisines.most_common(20)]
a,b=map(list,zip(*top_10_cuisines))
plt.figure(figsize=(8,6))
sns.barplot(b,a,palette="Spectral")
plt.title("Most famous cuisines in Bangalore")


# * North indian, Chinese, and South Indian are the top three cuisines in the Bangalore

# * ### What are the top 10 voted restaurants chain ?

# In[ ]:


data["votes"].describe()
b=data[['name','votes']].groupby("name").agg('mean').sort_values(by="votes",ascending=False)[:10]
plt.figure(figsize=(9,7))
sns.barplot(x=b.votes,y=b.index)


# ### How about the rate distribution?

# In[ ]:


rate=data['rate'].dropna()
rate=rate[~rate.str.contains(r'[NEW-]')]

#print(rate.map(type).value_counts()) # to check what kind of data type in this column

rate=rate.apply(lambda x: float(x.replace('/5','')))

plt.figure(figsize=(7,5))
sns.distplot(rate,bins=25)


# ### Does the rate corresponds to votes?

# In[ ]:


new=pd.DataFrame({"votes":data['votes'],'rate':rate})
new=new.dropna()
plt.figure(figsize=(10,7))
new.plot.scatter(x='rate',y='votes',color='b')    
plt.show()


# ### How much does it cost for eating in Bangalore?

# In[ ]:


data["approx_cost(for two people)"].map(type).value_counts()
cost=data["approx_cost(for two people)"].dropna()
cost=cost.apply(lambda x: int(x.replace(",","")))

plt.figure(figsize=(7,5))
sns.distplot(cost)
plt.ylabel("Freqency")
print ('The average cost for eating in Bangalore restaurants is {:.2f}'.format(cost.mean()))


# ### How many restaurants accept online oder?

# In[ ]:


oo=data['online_order'].value_counts().reset_index()
plt.figure(figsize=(7,5))
plt.pie(oo["online_order"],labels=oo['index'],autopct="%.2f%%")
plt.title('Online order accpeted or not')
plt.show()

