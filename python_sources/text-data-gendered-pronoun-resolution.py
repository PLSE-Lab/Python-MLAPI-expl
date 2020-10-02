#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
# Any results you write to the current directory are saved as output.


# In[ ]:


'''Contents
    Basic feature extraction using text data
        Number of words
        Number of characters
        Average word length
        Number of stopwords
        Number of numerics
        Number of uppercase words
    Basic Text Pre-processing of text data
        Lower casing
        Punctuation removal
        Stopwords removal
        Frequent words removal
        Rare words removal
        Spelling correction
'''


# In[ ]:


#Load GAP Coreference Data
gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", delimiter='\t')
gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')
gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')


# In[ ]:


#Load Competition Data
test_stage_1 = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')
sample_submission_stage_1 = pd.read_csv('../input/sample_submission_stage_1.csv')


# In[ ]:


test_stage_1.head()


# In[ ]:


sample_submission_stage_1.head()


# In[ ]:


print(test_stage_1.columns)


# In[ ]:


print(sample_submission_stage_1.columns)


# In[ ]:


test_stage_1.Pronoun.head()


# In[ ]:


test_stage_1.shape


# In[ ]:


test_stage_1.isna().sum()


# In[ ]:


##Number of Words

test_stage_1['word_count'] = test_stage_1['Text'].apply(lambda x: len(str(x).split()))
test_stage_1[['Text','word_count']].head()


# In[ ]:


print('Maximum num_words in test stage',test_stage_1["word_count"].max())
print('Min num_words in test stage',test_stage_1["word_count"].min())


# In[ ]:


#Number of characters

test_stage_1['char_count'] = test_stage_1['Text'].str.len() ## this also includes spaces
test_stage_1[['Text','char_count']].head()


# In[ ]:


#Average Word Length

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

test_stage_1['avg_word'] = test_stage_1['Text'].apply(lambda x: avg_word(x))
test_stage_1[['Text','avg_word']].head()


# In[ ]:


#Number of stopwords

stop = stopwords.words('english')
test_stage_1['stopwords'] = test_stage_1['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
test_stage_1[['Text','stopwords']].head()


# In[ ]:


#Number of numerics

test_stage_1['numerics'] = test_stage_1['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
test_stage_1[['Text','numerics']].head()


# In[ ]:


# Number of Uppercase words

test_stage_1['upper'] = test_stage_1['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
test_stage_1[['Text','upper']].head()


# In[ ]:


#Lower case

test_stage_1['Text'] = test_stage_1['Text'].apply(lambda x: x.lower())
test_stage_1['Text'].head()


# In[ ]:


#Removing Punctuation

test_stage_1['Text'] = test_stage_1['Text'].str.replace('[^\w\s]','')
test_stage_1['Text'].head()


# In[ ]:


#Removal of Stop Words

test_stage_1['Text'] = test_stage_1['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test_stage_1['Text'].head()


# In[ ]:


#Common word removal

freq = pd.Series(' '.join(test_stage_1['Text']).split()).value_counts()[:10]
freq = list(freq.index)
test_stage_1['Text'] = test_stage_1['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
test_stage_1['Text'].head()


# In[ ]:


#Rare words removal

freq = pd.Series(' '.join(test_stage_1['Text']).split()).value_counts()[-10:]
freq = list(freq.index)
test_stage_1['Text'] = test_stage_1['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
test_stage_1['Text'].head()


# In[ ]:


#Spelling correction
# train_df['question_text'] = train_df['question_text'].apply(lambda x: str(TextBlob(x).correct()))
# train_df['question_text'].head()

test_stage_1['Text'][:10].apply(lambda x: str(TextBlob(x).correct()))
test_stage_1['Text'].head()


# In[ ]:




