#!/usr/bin/env python
# coding: utf-8

# ** Overview **
# 
# This notebook will demo some common tricks used on feature engineering of numercie feature. Lastly, I will have a brief introduction of feature selection.
# 
# **Outline**
#     - Preface
#     1. Binning
#     2. Log, Power
#     3. Scaling and Normalization
#     4. Interaction Feature
#     5. Detail Feature
#     - Summary
#     - Introduction Feature Selection

# # Preface
# We will load the `yelp` dataset for this notebook, as you can see, the feature of this dataset is shown below.
# 
#     `'address','attributes','business_id', 'categories', 'city', 'hours', 'is_open', 'latitude', 'longitude', 'name', 'postal_code', 'review_count', 'stars', 'state'`
# 
# And we will focus on `review_count`.

# In[ ]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

with open('../input/yelp_academic_dataset_business.json','r') as f:
    raw_text = f.readlines()
    
business_dataset = pd.DataFrame([json.loads(s) for s in raw_text])

for col in business_dataset.columns:
    print(col)# The column of this table


# In[ ]:


print(business_dataset['review_count'].describe())


# In[ ]:


business_dataset.head(10)


# In[ ]:


fig,(ax1, ax2)=plt.subplots(1,2)
sns.set_style('whitegrid')

business_dataset['review_count'].hist(ax=ax1)
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Review Coount', fontsize=14)
ax1.set_ylabel('Occurence', fontsize=14)

business_dataset['review_count'].hist(ax=ax2)
# The axis scale type to apply. value : {"linear", "log", "symlog", "logit", ...}
ax2.set_yscale('log')
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Review Coount in log scale', fontsize=14)
_=ax2.set_ylabel('Occurence', fontsize=14)


# # 1. Binning
# Raw counts might span over several order of magnitude, and that might be a problem for some models. To deal with this problem, we can use `Binning`.
# ## 1. Fixed-width binning
# ## 2. Quantile binning
# 

# In[ ]:


def get_bins_by_fixed_length(fixed_length,display=False):
    floor = np.floor(business_dataset['review_count'].min())
    ceil = np.ceil(business_dataset['review_count'].max())
    bins = int(np.ceil((ceil-floor)/fixed_length))
    if display:
         print("Start from {} to {} with {} bins".format(floor,ceil,bins))
    return bins


# In[ ]:


bins = get_bins_by_fixed_length(10)
x = pd.cut(business_dataset['review_count'],bins,labels=False)


# In[ ]:


x2 = pd.cut(business_dataset['review_count'],100,labels=False).value_counts()


# In[ ]:


deciles = business_dataset['review_count'].quantile([i*0.1 for i in range(10)])
deciles


# In[ ]:


sns.set_style('whitegrid')
fig, ax = plt.subplots()
business_dataset['review_count'].hist(ax=ax, bins=100)
for pos in deciles:
    handle = plt.axvline(pos, color='r')
ax.legend([handle], ['deciles'], fontsize=14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(labelsize=14)
ax.set_xlabel('Review Count', fontsize=14)
ax.set_ylabel('Occurence', fontsize=14)


# In[ ]:


import numpy as np
small_counts = np.random.randint(0,100,(20,2))
large_counts = np.random.randint(1e5,1e10,(20,2))


# In[ ]:


mix = np.concatenate([small_counts,large_counts])


# In[ ]:


plt.scatter(mix[:,0],mix[:,1])


# # 2. Log power
# The log transformer is a powerful tool for dealing with positive number with heavy-tailed distribution.
# 

# In[ ]:





# In[ ]:


business_dataset['log_review_count'] = np.log(business_dataset['review_count']+1)


# In[ ]:


model_out_log = linear_model.LinearRegression()
model_with_log = linear_model.LinearRegression()


# In[ ]:


score_out_log = cross_val_score(model_out_log,business_dataset['review_count'].values.reshape(-1,1),business_dataset['stars'],cv=10)
score_with_log = cross_val_score(model_with_log,business_dataset['log_review_count'].values.reshape(-1,1),business_dataset['stars'],cv=10)


# In[ ]:


for title, score in zip(['With log','Without log'],[score_with_log,score_out_log]):
    print(title,'mean',score.mean(),'std',score.std())


# # 3. Scaling and Normalization

# # 4. Interaction Feature

# # 5. Detail Feature

# # Summary

# # Introduction Feature Selection
