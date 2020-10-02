#!/usr/bin/env python
# coding: utf-8

# ## TEST ##

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Reading data
dataset = pd.read_csv('../input/songdata.csv')

# Transform into TFIDF
corpus = dataset['text'].tolist()
tfidf = TfidfVectorizer(min_df=1, smooth_idf=True)
tfidf.fit_transform(corpus)

#
for artist, data in dataset.groupby('artist'):
    feat = np.sum(tfidf.transform(data['text']), axis=0)
    sort_idx = np.argsort(feat)
    print(sort_idx.tolist())
    break

