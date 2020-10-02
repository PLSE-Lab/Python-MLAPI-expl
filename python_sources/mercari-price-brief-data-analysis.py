#!/usr/bin/env python
# coding: utf-8

# **About the competition**
# 
# This is a kernel-only competition is hosted by Mercari, a Japan-based community-powered shopping app. 
# They provide user-inputted text descriptions of products and on the basis of this data, they aim to predict the sale price as it can vary depending upon a number of constraints. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# As easy as it could be, we are provided with simply a training and a testing tsv files for price prediction.
# Exploring the train.tsv file

# In[ ]:


train = pd.read_table('../input/train.tsv', engine='c')
print("(Rows, Columns) : ",train.shape)


# In[ ]:


train.head()


# **Some basic Analysis**
# 

# In[ ]:


for col in train:
    val=train[col].isnull().sum()*100.0/train.shape[0]
    if val>0.0:
        print("Missing values in",col,":",val)


# In[ ]:


price = train['price']
print(price.describe())
plt.figure(figsize=(12,12))
plt.scatter(range(train.shape[0]),np.sort(price.values))
plt.ylabel('Price', fontsize=12)
plt.xlabel('Index', fontsize=12)
plt.show()


# So the price starts with 0\$ and rises way to 2009\$ with the median being around just 17\$. 
# 
# Around 75 percent of the products have priced below 30$. 

# In[ ]:


plt.figure(figsize=(12,12))
sns.countplot(x="item_condition_id", data=train, palette="Greens_d")
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# There are five unique values of item_condition_id with "1" being of highest frequency and "4", "5" being the rarer ones.

# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x='item_condition_id', y='price', data=train[train['price']<100])
plt.ylabel('Price', fontsize=12)
plt.xlabel('Item Condition', fontsize=12)
plt.show()


# Overall, we can observe a slight decrease in prices of the products as its condition_id move from 1 to 4. In case of condition_id 5, the price distribution is wider, that might be because of its very less frequency (as observed in the previous graph), there are not enough data points to compare with other categories.

# In[ ]:


plt.figure(figsize=(20, 15))
plt.hist(train[train['shipping']==1]['price'],normed=True, bins=100, range=[1,250],alpha=0.6,color=['crimson'])
plt.hist(train[train['shipping']==0]['price'], normed=True,alpha=0.6,bins=100, range=[1,250],color=['blue'])
plt.xlabel('Price', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()


# We can see that prices are lower in case of free shipping.

# In[ ]:


brand = train['brand_name']
brand.describe()
from collections import Counter
c = Counter(brand.dropna())
print(c.most_common(10))


# There are 4809 unique brands present in the data with PINK with the highest product count. We also know that about 42 percent of products don't even have a specified brand.

# **WordCloud**

# In[ ]:


from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train['name']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# In the above wordcloud for names of the products, we can observe the frequent words (bigger) mostly consist of Brand names only. Some interesting phrases sellers often include in the name of the products:
# 1. "Free Ship/Shipping", "Brand New": So yes, that does attract people.
# 2. "VS": Cool!
# 3. "iPhone 6s" !?
# 4. "Vintage"
# 5. We also observe some clothing related phrases like "long sleeve", "size small" etc.
# 6. Color description like "Blue", "Grey", "White" etc.

# Talking about the description, we already know that some of the products weren't having one.

# In[ ]:


cloud = WordCloud(width=1440, height=1080).generate(" ".join(str(v) for v in train['item_description']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# **Category**
# 
# Missing percent = 0.43
# Now the categories are described in 3 levels at most.

# In[ ]:


def cat_name(cat):
    try:
        return cat.split('/')
    except:
        return np.nan, np.nan, np.nan

train['main'], train['l1'], train['l2'] = zip(*train['category_name'].apply(cat_name))


# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x='main', y='price', data=train[train['price']<100])
plt.ylabel('Price', fontsize=12)
plt.xlabel('Main Category', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


train.groupby(["main"])["main"].count().reset_index(name='count').sort_values(['count'], ascending=False).head(10)


# This is the overall availability of products in these main categories. Pretty Clear!

# In[ ]:


cat_count = train.groupby(["main","l1"])["l1"].count().reset_index(name='count')
cat_count.sort_values(['count'], ascending=False).head(10)


# So maximum products belong to Beauty->Makeup, Women->Athletic Apparel and Women->Tops & Blouses.

# In[ ]:


prices = train.groupby(['main','l1'])['price'].mean().reset_index(name='mean')
prices.sort_values(['mean'], ascending=False).head(10)


# These are some most expensive categories. 

# In[ ]:


prices.sort_values(['mean']).head(10)


# Similarly, categories with lowest mean prices are mainly Handmade or Home.

# In[ ]:


price_count = prices.merge(cat_count,left_on=['main','l1'],  right_on = ['main','l1'],how='inner')
price_count = price_count.sort_values(['count'], ascending=False).head(50)
plt.figure(figsize=(20, 15))
plt.barh(range(0,len(price_count)), price_count['mean'], align='center', alpha=0.5)
plt.xlabel('Price', fontsize=15)
plt.yticks(range(0,len(price_count)), price_count.l1, fontsize=15)
plt.ylabel('Level 1 category', fontsize=15)


# These are mean prices for some most popular level 1 product categories.

# For level 2 category:

# In[ ]:


prices = train.groupby(['main','l1','l2'])['price'].mean().reset_index(name='mean')
prices.sort_values(['mean'], ascending=False).head(10)

