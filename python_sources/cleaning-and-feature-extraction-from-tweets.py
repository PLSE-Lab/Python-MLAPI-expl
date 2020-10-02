#!/usr/bin/env python
# coding: utf-8

# # Cleaning and feature extraction from tweets
# 
# This notebook demonstrate a few ideais to clean tweets and also extract features of it without having to wait more than seconds to extract those features (no loops or messy sintax)

# In[ ]:


import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import groupby

from nltk.corpus import stopwords

re_url = r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'


# # Functions

# In[ ]:


def clean_text(text):
    '''Make text lowercase, remove reply, remove text in square brackets, remove links, remove user mention,
    remove punctuation, remove numbers and remove words containing numbers.'''
        
    text = text.lower()
    text = re.sub('^rt', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(re_url, '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('@\w+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

def get_consecutive_chars(text):
    ''' Count how many consecutive chars, consecutive upper chars and consecutive punctuation'''
    result = [(label, sum(1 for _ in group)) for label, group in groupby(text)]
    
    consecutive_chars = 0
    consecutive_chars_upper = 0
    consecutive_punctuations = 0
    
    for i in result:
        if i[1] > 1:
            if i[0] in string.punctuation:
                consecutive_punctuations += i[1]
            elif i[0].upper() == i[0]:
                consecutive_chars_upper += i[1]
            else:
                consecutive_chars += i[1]
                
    return {
        'qtd_consecutive_chars' : consecutive_chars,
        'qtd_consecutive_chars_upper': consecutive_chars_upper,
        'qtd_consecutive_punctuation' : consecutive_punctuations,
    }


# In[ ]:


clean_text('Test 123 of the function clean_text!! https://fake_url/2020')


# In[ ]:


get_consecutive_chars('test of the function get_consecutive_chars!! lool...')


# # Read dataset

# In[ ]:


# Read datasets
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

# Store idx for train and test
idx_train = df_train['id'].values
idx_test = df_test['id'].values


# I'll treat both dataframes as one to do cleaning only once, i should split it later
df_full = pd.concat([df_train, df_test], sort=False)


# # Clean tweets

# In[ ]:


stop_words = stopwords.words('english')

# Apply cleaning function
df_full['text_cleaned'] = df_full['text'].apply(lambda x: clean_text(x))

# Remove stop words
df_full['text_cleaned'] = df_full['text_cleaned'].str.split()     .apply(lambda x: [word for word in x if word not in stop_words])     .apply(lambda x: ' '.join(x))


# # Extract features

# ## Extract quantity values

# In[ ]:


df_full['qnt_words'] = df_full['text_cleaned'].str.split().apply(lambda x : len(x))
df_full['qnt_unique_words'] = df_full['text_cleaned'].str.split().apply(lambda x : len(set(x)))
df_full['qnt_chars'] = df_full['text'].str.len()
df_full['qnt_hashtags'] = df_full['text'].str.findall(r'#(\w+)').apply(lambda x : len(x))
df_full['qnt_user_mention'] = df_full['text'].str.findall(r'@(\w+)').apply(lambda x : len(x))
df_full['qnt_punctuation'] = df_full['text'].str.replace(r'[\w\s#]+', '').apply(lambda x : len(x))
df_full['qnt_urls'] = df_full['text'].str.findall(re_url).apply(lambda x : len(x))
df_full['mean_chars_words'] = df_full['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]))

df_full['qnt_stop_words'] = df_full['text'].str.split()     .apply(lambda x: len([w for w in x if w.lower() in stop_words]))


# ## Text contains hashtags, user mentions, urls or punctuation

# In[ ]:


df_full['contains_hashtags'] = df_full['text'].str.findall(r'#(\w+)').apply(lambda x : 0 if len(x) == 0 else 1)
df_full['contains_user_mention'] = df_full['text'].str.findall(r'@(\w+)').apply(lambda x : 0 if len(x) == 0 else 1)
df_full['contains_punctuation'] = df_full['text'].str.replace(r'[\w\s#]+', '').apply(lambda x : 0 if len(x) == 0 else 1)
df_full['contains_urls'] = df_full['text'].str.findall(re_url).apply(lambda x : len(x))

df_full['is_reply'] = df_full['text'].str.startswith('RT') + 0


# ## How many consecutive chars or punctuation the text has

# In[ ]:


df_consecutive = df_full['text'].apply(lambda x : pd.Series(get_consecutive_chars(x)))

for col in df_consecutive.columns:
    df_full[col] = df_consecutive[col]


# ## Contains Hashtags used previously in disaster tweets
# 
# Create column that contains the quantity of hashtags that were used in real disaster in other tweets

# In[ ]:


list_hashtags_disasters = []
for hashtags in df_full[(df_full['target'] == 1) & (df_full['qnt_hashtags'] > 0)]['text'].str.findall(r'(?<=#)(.*?)(?=\s)').values:
    for hashtag in hashtags:
        if hashtag.lower() not in list_hashtags_disasters and not hashtag.isdigit():
            if len(hashtag) > 2:
                list_hashtags_disasters.append(hashtag.lower())

list_hashtags_disasters.sort()


# In[ ]:


df_full['qtd_hashtags_used_in_disasters'] = df_full['text']    .str.findall(r'(?<=#)(.*?)(?=\s)')    .apply(lambda x : len([w for w in x if w.lower() in list_hashtags_disasters]))


# ## Check up of our features

# In[ ]:


df_full.columns


# In[ ]:


df_full.head()


# # Split cleaned dataframe in train and test

# In[ ]:


df_train = df_full[df_full['id'].isin(idx_train)]
df_test = df_full[df_full['id'].isin(idx_test)]

