#!/usr/bin/env python
# coding: utf-8

# Hello again! 
# 
# This notebook follows from the one I did on the analysis of the "manager_id" feature in the dataset (https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/do-managers-matter-some-insights-on-manager-id).
# 
# This time I try to get some insights on "building_id" and the "interest_level" for different buildings.
# 
# Again, let me know what you think about it and "up it" if you liked it.

# In[ ]:


# let's load the usual packages first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ... and get the data...
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
print(train_df.shape, test_df.shape)


# To start with let's see how many different buildings we have on both datasets.

# In[ ]:


bld_train_list = train_df.building_id.unique()
bld_test_list = test_df.building_id.unique()
print("Train : {0}".format(len(bld_train_list)))
print("Test  : {0}".format(len(bld_test_list)))
print("\nAverage entries for building in Train : {0: 1.3f}".format(len(train_df)/len(bld_train_list)))
print("Average entries for building in Test  : {0: 1.3f}".format(len(test_df)/len(bld_test_list)))


# That's quite a lot of buildings, and we have on average only a handful of entries for each building. 
# 
# Actually there are even more buildings than those just reported here, since many building are grouped together under a single 'id'. This will result in even lower average figures of entries-per-building.
# 
# Let's see this by creating a dataframe with all the train and test buildings, including the number of entries that refer to them.

# In[ ]:


# create dataframe
temp1 = train_df.groupby('building_id').count().iloc[:,-1]
temp2 = test_df.groupby('building_id').count().iloc[:,-1]
df_buildings = pd.concat([temp1,temp2], axis = 1, join = 'outer')
df_buildings.columns = ['train_count','test_count']
print(df_buildings.head(10))


# See the building labeled as '0'?! What is this? An anthill with thousands of apartments, or a legion of unlabeled buildings?
# 
# **The entries for '0' account for a big chunk of the whole datasets, over 15%**. 
# 
# To be more precise:

# In[ ]:


print("{0:2.3f}%".format(df_buildings.loc['0','train_count']/len(train_df)*100))
print("{0:2.3f}%".format(df_buildings.loc['0','test_count']/len(test_df)*100))


# It's easy to see that '0' stands for a group of several other buildings. See, they are allover NYC...

# In[ ]:


ixes0 = (train_df.building_id == '0')
plt.scatter(train_df[ixes0]['longitude'],train_df[ixes0]['latitude'])
plt.gca().set_xlim(-74.15,-73.7)
plt.gca().set_ylim(40.5,41)


# But how many of them there are? 
# 
# We can group by "longitude" and "latitude", hoping that these values are precises and able to pinpoint a unique building in the map. Alternatively, we could group by 'street_address' or 'display_address'.

# In[ ]:


gby0 = train_df[ixes0].groupby(['longitude','latitude'])
print("Total number of unique buildings ['longitude','latitude']: {0}".format(len(gby0.count())))
gby0a = train_df[ixes0].groupby(['display_address'])
print("Total number of unique buildings ['display_address']: {0}".format(len(gby0a.count())))
gby0b = train_df[ixes0].groupby(['street_address'])
print("Total number of unique buildings ['street_address']: {0}".format(len(gby0b.count())))


# The first option seems more promising since it reduces the number of buildings and it avoids problems due to different, wrong or incomplete addresses, such as:

# In[ ]:


print('Same building, different addresses (one more space character)\n')
print(gby0['display_address'].value_counts().loc[-73.9586,40.7704])
print('\n\nSame addresses (incomplete), different buildings\n')
print(gby0['display_address'].value_counts().loc[-74.0122,[40.7029,40.7040]])


# Anyway, grouping by lon/lat may also give some problems... (or spot some errors in the data entry of the addresses?)

# In[ ]:


print('Same coordinates, but different addresses (?!)\n')
print(gby0['street_address'].value_counts().loc[-74.0134,40.7056])


# ... well, let's forget about this and have a look at the locations labeled as '0' with most entries..

# In[ ]:


gby0['street_address'].value_counts().sort_values(ascending = False).head(10)


# Indeed it seems like some of these buildings have enough entries to deserve their own 'building_id'!
# 
# However, fragmentation still reigns within 'building_id' == 0' as most locations seem to have only a few entries.

# In[ ]:


gby0['street_address'].value_counts().sort_values(ascending = False).plot(kind = 'hist', bins = 50)


# Let's complete this first part by looking at the averaged interest levels for group '0'...

# In[ ]:


print('Average interest levels for group 0')
print(train_df[ixes0]['interest_level'].value_counts()/sum(ixes0))
print('\nOverall average')
print(train_df['interest_level'].value_counts()/len(train_df))


# Oh...!!! The entries for 'building_id' = 0 **are substantially less interesting than the average!** 
# 
# One could always say the interest level is 'low' and be correct 9 times out of 10!

# This should be further investigated, I reckon, but let's now forget about 'group 0' and consider the buildings with most entries having proper ids.
# 
# In particular, let's focus on the top 100 (according to the training dataset).

# In[ ]:


print(df_buildings.drop('0').sort_values(by = 'train_count',ascending = False).head(100))


# They have entries in both datasets, and these numbers seem fairly correlated.

# In[ ]:


df_buildings.drop('0').sort_values(by = 'train_count',ascending = False).head(100).corr()


# In[ ]:


temp = df_buildings.drop('0').sort_values(by = 'train_count',ascending = False).head(100)
plt.scatter(temp['train_count'],temp['test_count'])


# These top 100 account for a good 20% of the whole training dataset (similar figures for the test dataset)

# In[ ]:


temp = df_buildings.drop('0')['train_count'].sort_values(ascending = False).head(100)
temp = pd.concat([temp,temp.cumsum()/df_buildings['train_count'].sum()*100], axis = 1).reset_index()
temp.columns = ['building_id','count','cumulative percentage']
print(temp.head())
print('----------------')
print('----------------')
print(temp.tail())


# Let us consider now the average interest levels for these 100 buildings. Dummies are used for the three levels.

# In[ ]:


bld_list = temp['building_id']
ixes = train_df.building_id.isin(bld_list)
df100 = train_df[ixes][['building_id','interest_level']]
interest_dummies = pd.get_dummies(df100.interest_level)
df100 = pd.concat([df100,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)
print(df100.head())


# In[ ]:


# compute means and concat with count
gby = pd.concat([df100.groupby('building_id').mean(),df100.groupby('building_id').count()], axis = 1).iloc[:,:-2]
gby.columns = ['low','medium','high','count']
print(gby.sort_values(by = 'count', ascending = False).head())


# In[ ]:


# let's visualize them with a stacked bar chart
gby.sort_values(by = 'count', ascending = False).drop('count', axis = 1).plot(kind = 'bar', stacked = True, figsize = (15,5))
plt.figure()
gby.sort_values(by = 'count', ascending = False)['count'].plot(kind = 'bar', figsize = (15,5))


# I see more homogeneous interest levels with respect to my previous analysis on 'manager_id'  (https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/do-managers-matter-some-insights-on-manager-id).
# 
# As done in that notebook for the manager' skill, we can compute now an "average" measure of interest by assigning 0 points for "lows", 1 for "mediums" and 2 for "highs".
# 
# The distribution of this variable is less skewed than the one we obtained for the manager skill and it has less extreme values, suggesting more homogeneity in the interest levels across different buildings.

# In[ ]:


gby['avg_interest'] = gby['medium']*1 + gby['high']*2 

print("Top performers")
print(gby.sort_values(by = 'avg_interest', ascending = False).reset_index().head())
print("\nWorst performers")
print(gby.sort_values(by = 'avg_interest', ascending = False).reset_index().tail())


# In[ ]:


gby.avg_interest.plot(kind = 'hist')
print(gby.mean().drop('count'))


# That's all for now folks!
