#!/usr/bin/env python
# coding: utf-8

# <h2>Introduction</h2>
# First let's explore a raw data a little bit, check the distributions, look for outliers and plan for future feature extraction. Start with importing neccessary libraries:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's see how the training data and the test data look:

# In[ ]:


train = pd.read_json('../input/train.json')
test  = pd.read_json('../input/test.json')
print(f'Training set is of size:{train.shape}')
print(f'Test set is of size:{test.shape}')
train.head(1)


# We will use a binary normalized histogram in further investigations, so let's define the method once to reuse it later:

# In[ ]:


def PlotNormHist(data, axes, binaryFeat):
    param = [False, True]
    for i,cur_ax in enumerate(axes):
        cur_data = data[data[binaryFeat]==param[i]]
        int_level = cur_data['interest_level'].value_counts()
        int_level = int_level/sum(int_level)
        sns.barplot(int_level.index, int_level.values, alpha=0.8,
                    order=['low','medium','high'], ax=cur_ax)
        cur_ax.set_xlabel(param[i], fontsize=15)
        cur_ax.set_ylim(bottom=0, top=1)
        cur_ax.grid()


# <h2>Photos</h2>
# At this stage we will try not to use the images at all, but it can be usefull to check if the total number of images per specific listing can help us in prediction. Let's create a new numerical column nPhotos and threshold the maximal number of photos to 10 to avoid strange outliers:

# In[ ]:


train['nPhotos'] = train['photos'].apply(lambda x: min(10, len(x)))


# Now let's check the distribution over the "interest level", using the "violin" plot:

# In[ ]:


plt.figure(figsize=(10,5))
sns.violinplot(x='interest_level', y='nPhotos', data=train, order=['low','medium','high'])
plt.xlabel('# Interest Level', fontsize=12)
plt.ylabel('# of Photos', fontsize=12)
plt.grid()
plt.show()


# We can see that a lack of photos significantly decreases the chances of the listing to be popular as we would expect.

# <h2>Description</h2>
# For now, we won't use the provided 'description' field. We can assume that written keywords in this field can play significant role in the prediction, so we want at least create a new column that will have a binary indication for every listing - True if the listing had some sort of description or False otherwise. Let's check the distribution over the "interest level" using the simple countplot:

# In[ ]:


train['hasDesc'] = train['description'].apply(lambda x: len(x.strip())!=0)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
PlotNormHist(train, axes, 'hasDesc')


# It is obvious that a lack of description lowers the chances of the listing to become popular.

# <h2>Geographical Distribution</h2>
# We can assume that geographical location of the listing will have a significant influence on the interest levels for this listing. Specifically we can think of a distribution of neighborhoods where some of them would be "prestigiuos" and other less. Let's try and plot the listings on map projection:

# In[ ]:


long_llim = np.percentile(train.longitude.values, 1)
long_ulimit = np.percentile(train.longitude.values, 99)
lat_llim = np.percentile(train.latitude.values, 1)
lat_ulimit = np.percentile(train.latitude.values, 99)
train = train[(train['longitude']>long_llim) & (train['longitude']<long_ulimit) & 
              (train['latitude']>lat_llim) & (train['latitude']<lat_ulimit)]
lats = list(train['latitude'])
lons = list(train['longitude'])


# In[ ]:


fig = plt.figure(figsize=(15, 15))
m = Basemap(projection='merc',llcrnrlat=min(lats),urcrnrlat=max(lats),            llcrnrlon=min(lons),urcrnrlon=max(lons), resolution='h')
x, y = m(lons,lats)
sns.scatterplot(x, y, hue=train['interest_level'], style=train['interest_level'])


# We can see that the density of the classes indeed differs, depending on the geographical location. There is no "hard" separator, but the usefullnes of the {lat, long} feature is obviuos. It may be nice to try and run a KNN classification and maybe create an additional feature that measures the distance of the listing from the center.

# <h2>Price:</h2>
# Let's see the distribution of the prices:
# 

# In[ ]:


ulimit = np.percentile(train.price.values, 99)
train['price'][train['price']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.grid()
plt.show()


# How does the price varies over the interest categories:

# In[ ]:


plt.figure(figsize=(10,5))
sns.violinplot(x='interest_level', y='price', data=train, order=['low','medium','high'])
plt.xlabel('# Interest Level', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid()
plt.show()


# <h2>Features:</h2>
# Let's check now the column of the categorical feature entries:
# We will count the occurencies of every feature:

# In[ ]:


feat_dict = {}
for ind, row in train.iterrows():
    for f in row['features']:
        f = f.lower().replace('-', '')
        if f in feat_dict:
            feat_dict[f] += 1
        else:
            feat_dict[f] = 1 


# Let's see the most common features:

# In[ ]:


new_feat_dict = {}
for k,v in feat_dict.items():
    if v>50: new_feat_dict[k] = v  
new_feat_dict.keys()


# Some common features can have different variations, for example {'parking', 'onsite garage','garage'} are basically the same and can be very significant (deriving from our daily experience trying to find parking).
# Let's check for example someof the features for significance on interest level:

# In[ ]:


def CreateCategFeat(data, features_list):
    f_dict = {'hasParking':['parking', 'garage'], 'hasGym':['gym', 'fitness', 'health club'],
              'hasPool':['swimming pool', 'pool'], 'noFee':['no fee', "no broker's fees"],
              'hasElevator':['elevator'], 'hasGarden':['garden', 'patio', 'outdoor space'],
              'isFurnished': ['furnished', 'fully  equipped'], 
              'reducedFee':['reduced fee', 'low fee'],
              'hasAC':['air conditioning', 'central a/c', 'a/c', 'central air', 'central ac'],
              'hasRoof':['roof', 'sundeck', 'private deck', 'deck'],
              'petFriendly':['pets allowed', 'pet friendly', 'dogs allowed', 'cats allowed'],
              'shareable':['shares ok'], 'freeMonth':['month free'],
              'utilIncluded':['utilities included']}
    for feature in features_list:
        data[feature] = False
        for ind, row in data.iterrows():
            for f in row['features']:
                f = f.lower().replace('-', '')
                if any(e in f for e in f_dict[feature]):
                    data.at[ind, feature]= True     
cat_features = ['hasParking', 'hasGym', 'hasPool', 'noFee', 'hasElevator',
                'hasGarden', 'isFurnished', 'reducedFee', 'hasAC', 'hasRoof',
                'petFriendly', 'shareable', 'freeMonth', 'utilIncluded']
CreateCategFeat(train, cat_features)


# In[ ]:


for cur_feature in cat_features:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))
    PlotNormHist(train, axes, cur_feature)
    fig.suptitle(cur_feature, fontsize=16)


# In[ ]:


import datetime
train['created'] = pd.to_datetime(train['created'])
train['month']   = train['created'].dt.month
plt.figure(figsize=(8,6))
sns.countplot(x='month', hue='interest_level', data=train, hue_order=['low','medium','high'])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('month', fontsize=12)
plt.grid()


# In[ ]:


train['weekday'] = train['created'].apply(lambda x: x.weekday())
plt.figure(figsize=(8,6))
sns.countplot(x='weekday', hue='interest_level', data=train, hue_order=['low','medium','high'])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Weekday', fontsize=12)
plt.grid()


# <h2>Summary:</h2>
# People are interested in apartment listings that have **description** and **photos**. Apartments with extreme outlied **prices** have less interest levels. 
# Geographical location has significant value in terms of **latitude**, **longitude**.
# People look for *furnished* , *air-conditioned* apartments that don't have an additional *fee* , have a *reduced fee* or have other $ perks as *free month* or *utilities included*.
# Other features, exspecially *parking, gym, elevator* or *garden* can be important, but may be masked by the influence of the price.
# In addition it looks like *shareable apartments* are popular, since more young population looks for roommates apartments.
# 
