#!/usr/bin/env python
# coding: utf-8

# # Combining Data from TED Talks on Kaggle
# 
# **Aim: **
# 
# 1. Convert transcript(s) downloaded using [YouTube-dl](http://rg3.github.io/youtube-dl/) to csv format, and providing a dataset compatible for Kaggle (**this is uploaded as the files in Ted-talks-transcript**)

# In[ ]:


import pandas as pd
import os
import json


# In[ ]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# # Language codes seen in the dataset:
# 
# large number of language codes used, multiple similar but different languages. e.g. Mandrain: zh-CN, zh-TW,...

# In[ ]:


df_info = pd.read_csv('../input/ted-talks-transcript/tedDirector.stats.csv',
                      names=['lang','number'],sep='\t')


# In[ ]:


LANGUAGES = list(df_info.lang)
print(sorted(LANGUAGES))


# In[ ]:


print('Total number of langauges :', len(LANGUAGES))


# # Appending video ID into kaggle's data:
# 
# This section describes how the video IDs (from youtube) are codified in [../input/ted-talks-transcript/ted_metadata_kaggle.csv](../input/ted-talks-transcript/ted_metadata_kaggle.csv)

# In[ ]:


# Metadata from YOUTUBE
df_metadata = pd.read_csv('../input/ted-talks-transcript/ted_metadata_youtube.csv')
df_metadata.info()


# In[ ]:


df_metadata.head(3)


# In[ ]:


# Sample the titles:
df_metadata.fulltitle.values


# In[ ]:


df_kaggle = pd.read_csv('../input/ted-talks-transcript/ted_metadata_kaggle.csv')
df_kaggle.info()


# In[ ]:


df_kaggle.head(3)


# ## Preprocess the `titles` from both dataset and fuzzy match
# 
# since the number of ted videos we scrapped is less than that in df_kaggle, 
# we will expect some of the entries in kaggle to not have any youtube vidID
# Fuzzy matching:
# 1) Compare the titles in df_kaggle and df_meta -> most probably will fail
# 2) Count the proportion of matching tokens in the title; and assign ID based on the highest proportion

# In[ ]:


import re
import nltk


# In[ ]:


import string
from nltk.corpus import stopwords # Common stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

### Functions for simple preprocessing of summary:
sw = stopwords.words('english')
sw.extend(list(string.punctuation))
stop = set(sw)

# tokenizing the sentences; this uses punkt tokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokenize = lambda x : tokenizer.tokenize(x)

# apply stopping, and remove tokens that have length of 1
removeSW = lambda x : list([t.lower() for t in x if t.lower() not in stop and len(t) > 1 and t.isalpha()])

# Lemmatizing
lemmatizer = WordNetLemmatizer()
lemmify = lambda x : [lemmatizer.lemmatize(t) for t in x]

preprocess = lambda x: lemmify(removeSW(tokenize(x)))


# In[ ]:


# Approach 2): lower case both df for comparison:
df_kaggle['title_compare'] = df_kaggle['name'].apply(lambda x:preprocess(x))
df_metadata['fulltitle_compare'] = df_metadata['fulltitle'].apply(lambda x :preprocess(x))


# #### `titles` after preprocessing

# In[ ]:


df_metadata['fulltitle_compare'].iloc[0:5].values


# In[ ]:


df_kaggle['title_compare'].iloc[0:5].values


# #### Fuzzy matching between both dataset
# 
# Using the titles, I used a simplified way of matching the videos IDs from YouTube metadata (which is also the labels of the transcripts) to the metadata from TED.com

# In[ ]:


metadata_compare = df_metadata[['fulltitle_compare', 'id']] # small dataset for comparing
metadata_compare.set_index('id',inplace=True)


# In[ ]:


def find_match(toks_kaggle, threshold=0.8):
    #     only accept the vidID if the threshold is greater than or equal 0.8   
    compare = metadata_compare.copy()
    compare['ratings'] = compare['fulltitle_compare'].apply(lambda meta: compute_score(toks_kaggle, meta))
    vidID = compare.ratings.idxmax()
    
    if compare.ratings.max() >= threshold:
#         logging.info("FOUND: {}".format(compare.loc[vidID]))
#         logging.info("-----ACCEPT----")
        return vidID
    elif compare.ratings.max() >= .7:
        # Use this to check if the threshold is ``enough''
        logging.info("KAGGLE TITLE: {}".format(toks_kaggle))
        logging.info("FOUND: {} ({})".format(compare.loc[vidID].fulltitle_compare, compare.loc[vidID].ratings))
        logging.info("-----REJECT----")
        return ""
    else:
        return ""

def compute_score(kaggle, meta):
    total= len(kaggle)
    count = 0
    for toks in meta:
        if toks in kaggle: 
            count += 1
    return count/total


# In[ ]:


# This would generate the youtube Video ID for the kaggle dataset
df_kaggle['vidID_youtube'] = df_kaggle['title_compare'].apply(lambda x : find_match(x))


# In[ ]:


# Proportion of kaggle dataset not labelled with any videoIDs:
print(list(df_kaggle.vidID_youtube.values).count(''), "/", len(df_kaggle), " = ", list(df_kaggle.vidID_youtube.values).count('')/len(df_kaggle))


# About 13.3% of the kaggle data is not succesfully mapped to the transcript dataset from YouTube!

# #### ?? How can we improve this;

# In[ ]:


## These are then save for this dataseet.
# df_kaggle.to_csv('./ted_metadata_kaggle.csv')
# df_metadata.to_csv('./ted_metadata_youtube.csv')

