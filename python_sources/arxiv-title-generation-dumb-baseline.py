#!/usr/bin/env python
# coding: utf-8

# There's a subtle fault in the arrangement of this competition - 431 papers from the test set are also found in the training set. 
# 
# So a dumb baseline looks like this:
#  - a title of a duplicate paper form the training set if a duplicate is found
#  - first sentence from the abstract otherwise (but no more than 40 words)

# In[ ]:


import pickle
import string
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from nltk.util import ngrams


# In[ ]:


PATH_TO_DATA = Path('../input/title-generation/')


# In[ ]:


train_df = pd.read_csv(PATH_TO_DATA / 'train.csv')


# In[ ]:


test_df = pd.read_csv(PATH_TO_DATA / 'test.csv')


# In[ ]:


len(train_df), len(test_df)


# Removing duplicates in the training set.

# In[ ]:


train_df.drop_duplicates(inplace=True)
len(train_df)


# In[ ]:


train_df.head(2)


# In[ ]:


test_df.head(2)


# In[ ]:


train_abstracts = train_df['abstract'].str.lower()
test_abstracts = test_df['abstract'].str.lower()


# Finding duplicates

# In[ ]:


duplicate_ids = {}

for i, abstract in tqdm(enumerate(test_abstracts)):
    
    # a bit clumpsy, but pd.Series.str.contrains is not working for me
    inclusion_series = (abstract == train_abstracts)
    if inclusion_series.sum():
        test_id = i
        train_id = inclusion_series.idxmax()
        duplicate_ids[test_id] =  train_id
        
len(duplicate_ids)


# Let's pick up the first sentence from the abstract as a hypothesis for a title. But no more than 40 words.

# In[ ]:


train_df['title'].apply(lambda s: len(s.split())).describe()


# In[ ]:


def extract_first_sentence(text, max_words=40):
    return " ".join(text.strip().split('.')[0].split()[:max_words])


# In[ ]:


predicted_titles = test_df['abstract'].apply(extract_first_sentence)


# **Replace some titles with titles for duplicate papers**

# In[ ]:


for test_id, train_id in duplicate_ids.items():
    predicted_titles.loc[test_id] = train_df.loc[train_id, 'title']


# ## Submission
# 
# Reusing code form the published [LSTM baseline](https://www.kaggle.com/anastasiayanina/lstm-baseline).

# In[ ]:


submission_df = pd.DataFrame({'abstract': test_df['abstract'].values, 
                              'title': predicted_titles.values})
submission_df.to_csv('predicted_titles.csv', index=False)


# In[ ]:


def generate_csv(input_file='predicted_titles.csv',
                 output_file='submission.csv',
                 voc_file='../input/title-generation/vocs.pkl'):
    '''
    Generates file in format required for submitting result to Kaggle
    
    Parameters:
        input_file (str) : path to csv file with your predicted titles.
                           Should have two fields: abstract and title
        output_file (str) : path to output submission file
        voc_file (str) : path to voc.pkl file
    '''
    data = pd.read_csv(input_file)
    with open(voc_file, 'rb') as voc_file:
        vocs = pickle.load(voc_file)

    with open(output_file, 'w') as res_file:
        res_file.write('Id,Predict\n')
        
    output_idx = 0
    for row_idx, row in data.iterrows():
        trg = row['title']
        trg = trg.translate(str.maketrans('', '', string.punctuation)).lower().split()
        trg.extend(['_'.join(ngram) for ngram in list(ngrams(trg, 2)) + list(ngrams(trg, 3))])
        
        VOCAB_stoi = vocs[row_idx]
        trg_intersection = set(VOCAB_stoi.keys()).intersection(set(trg))
        trg_vec = np.zeros(len(VOCAB_stoi))    

        for word in trg_intersection:
            trg_vec[VOCAB_stoi[word]] = 1

        with open(output_file, 'a') as res_file:
            for is_word in trg_vec:
                res_file.write('{0},{1}\n'.format(output_idx, int(is_word)))
                output_idx += 1


generate_csv()

