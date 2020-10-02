#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
from collections import defaultdict


# In[ ]:


train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
combine = [train_df, test_df]


# Take a look at the data.

# In[ ]:


train_df.head()


# Ok, we have a "created" field which is a date. That will have to be at least formatted, maybe split up.
# 
# Let's see what we have for missing values:

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# No null values. What about empty values

# In[ ]:


train_df.describe()


# We have some zero values for bathrooms, bedrooms, latitude and longitude. Those are probably all missing values. 
# 
# Let's look at categorical variables:

# In[ ]:


combine[0].drop(['features', 'photos'], axis=1).describe(include=['O'])


# Ok, we have some columns whose values are lists. Those mess up the describe, so we had to remove them for now. We will definitely have to handle them, particularly the 'features' variable as intuitively it likely contains some highly correlated information.
# 
# For now let's look at what we got though. It doesn't tell us much. The most common building ID is '0'. That's likely just an 'other' category. There are a lot of empty descriptions, those may or may not be useful. A simple 'length of description' feature might be useful.
# 
# Let's take a look at that features column:

# In[ ]:


features = defaultdict(int)
for featureList in combine[0]['features']:
    for feature in featureList:
        features[feature.lower()] += 1
        
features = pd.DataFrame.from_dict(features, orient='index')
features.columns = ['featureCounts']
features.describe([.8,.85,.9,.95,.99])


# Ok, there are a lot of features, but they have a sort of zipfian look to their numbers.  Let's try to prune them.

# In[ ]:



features = features[features['featureCounts'] > features['featureCounts'].quantile(.95)]
features.describe()


# In[ ]:


features.head()


# That gives us 65 features which is more doable. If we have time we should come back and select them in a smarter manner including:
# 
#  - Synonym resolution. We pick the low-hanging fruit of case-sensitivity, but there is more that can be done such as prewar == pre-war
#  - Better better scoring function than simple counts

# In[ ]:


def lowerCaseList(l):
    return [f.lower() for f in l]

for dataset in combine:
    dataset['features'] = dataset['features'].map(lowerCaseList)
    
for featureLabel in features.axes[0]:
    for dataset in combine:
        pass

