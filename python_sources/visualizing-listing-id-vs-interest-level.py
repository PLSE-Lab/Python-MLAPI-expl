#!/usr/bin/env python
# coding: utf-8

# I've seen some discussion around the listing_id being a useful feature, so I've created this notebook to visualize the relationship.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_json('../input/train.json')
#train['listing_id'] -= train['listing_id'].min()

order = ['low', 'medium', 'high']
plt.figure(figsize=(8, 10))
plt.title("Listing ID vs Interest Level")
sns.stripplot(train['interest_level'],train['listing_id'],jitter=True, order=order)
plt.show()


# As we can see above, below a certain listing_id value, the interest drops off sharply, and very high listing_id values have no interest at all.
# 
# Looking at this across time, we see data is even more interesting:

# In[ ]:


train['created'] = pd.to_datetime(train['created'])
train['day_of_year'] = train['created'].dt.dayofyear

plt.figure(figsize=(13,10))
#plt.figure(figsize=(8, 10))
#plt.title("Listing ID vs Interest Level")
train['week_of_year'] = train['day_of_year'] // 7
sns.boxplot(train['week_of_year'], train['listing_id'], train['interest_level'], 
              hue_order=order)
plt.show()


# I can't say for certain if this is a true leak, or some feature embedded in the listing_id values.  But it is very interesting. 
