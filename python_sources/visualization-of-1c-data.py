#!/usr/bin/env python
# coding: utf-8

# ## Visualization Sandbox
# 
# This kernel will explore many important and useful relationships between items, categories of items and shop ids. present in train and test. Most of the people are already exploring the time series behavior of this data which will be quite useful in framing the validation stage for modeling and to make the lag features. In this kernel, I am going to explore other areas which I observed as useful and important. I have created this kernel for the demonstration purposes and thus, haven't changed the names from Russian to English.
# I still don't know how to make use of all this as featuers to feed within the model but, this is an open ended question which I am leaving for later stages. 
# 
# The information we are going to discuss in this Kernel are:
#     
#     1. Missing values exploration [for unique combinations in test].
#     2. Famous Shops
#     3. Famous Item Categories
#     4. Joint Plots of Item Categories and Shop Ids for train/test
#     5. Newly Introduced items per month.
#     6. Newly Introduced items for November, 2015.
#     7. Items sold out in shops
#     7. Item Categories vs Items for 3 year
#     8. Deciding the lag features

# In[54]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[55]:


train = pd.read_csv('../input/sales_train.csv.gz')
item_desc = pd.read_csv('../input/items.csv', low_memory=True)
test = pd.read_csv('../input/test.csv.gz', low_memory=True)
shops = pd.read_csv('../input/shops.csv')
item_cat = pd.read_csv('../input/item_categories.csv', low_memory=True)
train.head()


# In[56]:


#Remove the duplicate rows
print('Before drop train shape:', train.shape) 
train.drop_duplicates(subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'], keep='first', inplace=True) 
train.reset_index(drop=True, inplace=True) 
print('After drop train shape:', train.shape)


# ## Outliers and Missing Value exploration

# In[57]:


train.describe()


# In[58]:


from matplotlib import pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(ncols = 2, figsize = (20, 10))
sns.boxplot(x='item_price', data = train, ax = ax[0])
ax[0].set_title('Boxplot for Item price')
sns.boxplot(x='item_cnt_day', data = train, ax = ax[1])
ax[1].set_title('Boxplot for Item counts')
plt.show()


# From the above mentioned diagram, it's quite clear that the boxplot for item price and item couns are mainly concentrated to small values (especially item counts ~0). The outliers can be easily identified as the values far away from the rest of the values.
# For this kernel purpose, I am taking item price values > 100K and item counts >= 1000 as the outliers. Also,from the desription table the item_price has fallen in negative range which is truely can't be defined. So, we need to remove that as well.

# In[59]:


train = train.loc[(train.item_price < 100000) & (train.item_cnt_day <= 1000) & (train.item_price >=0)]


# In[60]:


# Perform the aggregation according to every month.

dfs = []
for month_block in train.date_block_num.unique():
    print('Making dataframe for: %s' %(month_block+1))
    df = pd.DataFrame(train[train['date_block_num'] == month_block].groupby(['shop_id', 'item_id'])['item_cnt_day'].sum())
    df.reset_index(inplace = True)
    df['date_block_num'] = month_block
    dfs.append(df)
    
df = pd.DataFrame()
for frame in dfs:
    df = pd.concat([df, frame], axis = 0)
train = df.copy()


# In[61]:


get_ipython().system('pip install missingno #check this package out for useful exploration of missing data in dataset.')


# In[66]:


import missingno as msno

def aggregate_cols(row):
    return str(int(row['shop_id'])) + str(int(row['item_id']))
train['agg_id'] = train.apply(lambda row: aggregate_cols(row), axis = 1)
test['agg_id'] = test.apply(lambda row: aggregate_cols(row), axis = 1)

dic = {}
ids = train.agg_id.unique().tolist()
for id  in ids:
    dic[id] = 0

test['agg_id'] = test['agg_id'].map(dic)


# In[65]:


plt.style.use('ggplot')
msno.bar(test)


# Above visualization gives quite a clear idea of the unique combinations that are present in test but not in train.  Almost 50% of the data that is present in the train but is present in the test. Thus, we need to engage this information in train to make validation strategy more robust and more accurate model. 

# ## Famous Shops
# The famous shops are defined according to frequency of people that visited that shop or in short the number of items that are being bought from that shop which more or less represent the similar scenario. This information will be helpful to locate the shops in main areas vs remote areas. [Atleast that's what I think]

# In[8]:


train = pd.merge(train, shops, on = 'shop_id', how = 'inner')
Z = dict(train['shop_name'].value_counts())
fig, ax = plt.subplots(1, figsize=(15, 5))
sns.stripplot(list(Z.keys()), list(Z.values()), ax = ax)
plt.xticks(rotation = 90)
plt.show()


# Although, shops name don't make much sense but, we see some clear pattern in the shops that are frequently visited, less frequently visited to not visited at all on the basis of their occurrence in the dataset. We can divide the shops in clusters to give it as a feature [Just a thought].

# ## Famous Item Categories
# 
# The similar graph is built to define which item category is more famous or most frequently bought by the customers [again this depends on a lot of factors like cost of the item].

# In[9]:


item_desc = pd.merge(item_desc, item_cat, how='inner', on='item_category_id')
train = pd.merge(train, item_desc[['item_id', 'item_category_name', 'item_category_id']], on = 'item_id', how = 'inner')

Z = dict(train['item_category_name'].value_counts())
fig, ax = plt.subplots(1, figsize=(18, 5))
sns.stripplot(list(Z.keys()), list(Z.values()), ax = ax, edgecolor='black', size=5)
plt.xticks(rotation = 90)
plt.title('Item Categories set according to Frequency')
plt.show()


# The visualization also gives quite a good information on what items are highly sold out and what items are not being sold at all. It looks like DVD and bluray CDs are among the highly sold out items which makes perfect sense as the items are less expensive than PCs and game consoles like XBOX. This information can also be embedded in the model in the form of popular vs non popular item lag [same reason as we are creating mean encodings for item price and item count month].

# ## JOINT PLOTS
# 
# Behold the power of joint plots. This will uncover the truth of train and test set pretty quick and will help us frame the robust validation strategy. You will find a lot of interesting patterns within the join plots which you can feed the model as extra features.

# In[10]:


sns.jointplot('shop_id', 'item_category_id', data = train, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()


# In[11]:


test = pd.merge(test, item_desc[['item_id', 'item_category_name', 'item_category_id']], on = 'item_id', how = 'inner')
sns.jointplot('shop_id', 'item_category_id', data = test, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()


# Ah ha! Look at the graphs. Organizers have given constant item_categories in each shop [which are 42 in test] and thus, the test graph looks much more consistent rather than train graph. I have rechecked it and we have exactly 5100 items present across each shop which we have to check for. This validates that the company not only gives the items that are being sold out but, also the items that will have zero sales. 

# In[12]:


test['date_block_num'] = 34
g = pd.concat([train[['shop_id', 'date_block_num']], test[['shop_id', 'date_block_num']]])
sns.jointplot('date_block_num', 'shop_id', data = g, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()


# This joint plot of shops across the date block numbers give us an idea of the behavior of the shops with the timeline. I found following observations to be quite useful:
# 
# 1. The 34th month is mostly dependent [perfectly coinciding] on previous month sales information. This means that the shops that are present in november are mostly present in previous months except a few. This gives somewhat idea about validation.
# 2. Some shops like 0, 1, 32, etc. had a very short time period with 1c company which mean either they were closed or they their agreement got over with the company.
# 3. Some shops are quite seasonal or they have 1c company as their backup arrangements. This gives the insight that they won't be their in test data.

# ## Newly Introduce items Per month
# 
# This section will give us some idea to understand the launching strategies of the companies in the market across each month. This helps us understand how many items should be introduced in the November time of 2015 which we will validate later. For items to get settled down, I am taking only last 2 years i.e. 2014 and 2015.

# In[13]:


df = train[train.date_block_num >= 12]
months = np.sort(df.date_block_num.unique())
new_items_introduced = [0] * 12
for i in range(12, len(months)):
    new_items_introduced[i%12] += len(np.setdiff1d(df[df.date_block_num == months[i]]['item_id'].unique(), df[df.date_block_num < months[i]]['item_id'].unique()))

#names = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']       
plt.figure(figsize = (15, 5))
plt.bar(np.arange(1,13), new_items_introduced)
plt.title('Items introduced over months')
plt.xticks(np.arange(1,13))
plt.show()
'''
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(new_items_introduced, labels=names)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
'''


# A couple of things are quite clear that companies are introducing major items just before the major holidays or after the major holidays to get them settled in the market. The major months that contributed a lot are, October and March where companies introduces a lot of items in the market. It may depends on many other factors which I am missing.

# ## Items sold out in shops [Hexagonal Binning]
# 
# This section will discover if certain items are present in shops or not. With this information, we can check which items are present in which shops and will be able to feed this information within the model. This information helps us to locate if the shops are accessible to customers or they are located in remote areas or they are small scale vs large scale shops. I am still in the process to caliberate this information to provide the right information into the model.

# In[14]:


fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
hb1 = ax.hexbin(train.shop_id, train.item_id, cmap = 'inferno')
plt.title('Hexagonal Binning for the items present in the shops')
plt.xticks(np.arange(0,60))
plt.xlabel('Shop Id')
plt.ylabel('Item Id')
plt.show()


# From the visualization a couple of things are quite clearer:
# 
# 1. Shops from 25 to 28, 31 and 54 are locate in main markets somewhere and are quite a large scale shops and thus, have almost all the items present in quite a large amount
# 2. Some shops like 0, 1, 8, 9 etc. have almost next to nothing which indicates that either these shops were open for some time and get closed afterwards or they are very small scale shops.
# 3. Shops like 2,3,4,10,12,etc.  have limited item ids which indicates they are quite specific in what they are selling at a time [May be a cd store or game store or gift shop].

# ## Item categories vs Items for 3 years

# In[15]:


fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
hb1 = ax.hexbin(item_desc.item_category_id, item_desc.item_id, cmap = 'ocean')
plt.title('Hexagonal Binning for the item categories vs items')
plt.xticks(np.arange(0,85))
plt.xlabel('Item Category Id')
plt.ylabel('Item Id')
plt.show()


# Hmm. This looks quite intriguing. Only handful of item categories have a lots of items present in them while other item categories are quite empty as compared to others. Thus, we can make 2 hypothesis that 
# 1. The item categories with much less denser area are quite rarer to find in the shops compared to ones with more denser areas.
# 2. The item categories with much less denser area release quite less but famous products [like game consoles] compared to other products [like game cds].

# ## Deciding the Lag Features
# 
# Lag features are proven to be most powerful and useful for these set of problem. But, how many periods should we take into account? We can take a more closer look in the bar graph in order to explore these questions.

# In[33]:


Z = train.groupby('date_block_num').agg({'item_cnt_day': sum}).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_cnt_day', ax = ax, palette="BrBG")
plt.title('Overall Item counts over the course of 3 years')
plt.show()


# The general trend suggests that, the item counts are decreasing over the time till a certain time (roughly around May or June). The graph also suggests that there is a seasonality of around 12 months which means that the trend tends to repetitive over the time. Thus, we can take the lag features of roughly around 5 to 6 time periods along with 12 for seasonality purposes.
