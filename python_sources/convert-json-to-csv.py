#!/usr/bin/env python
# coding: utf-8

# It seems that in the recent version of the dataset, the original csv file format was removed, and replaced by JSONs. This extremely simple script converts the json format into a single CSV file, which contains only the text and the associated rating. Feel free to use the output.

# In[ ]:


import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


# In[ ]:


data = {'stars': [], 'text': [], 'review_id': []}

with open('../input/yelp_academic_dataset_review.json') as f:
    for line in tqdm(f):
        review = json.loads(line)
        data['stars'].append(review['stars'])
        data['text'].append(review['text'])
        data['review_id'].append(review['review_id'])


# In[ ]:


df = pd.DataFrame(data)

print(df.shape)
df.head()


# In[ ]:


df['stars'] = df['stars'].astype('category')
df['text'] = df['text'].astype(str)
df['review_id'] = df['review_id'].astype(str)


# In[ ]:


df.shape


# In[ ]:


df.to_pickle('reviews.pkl')


# In[ ]:




