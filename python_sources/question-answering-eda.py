#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import json
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## File Description
# - `simplified-nq-train.jsonl` - the training data, in newline-delimited JSON format.
# - `simplified-nq-kaggle-test.jsonl` - the test data, in newline-delimited JSON format.
# - `sample_submission.csv` - a sample submission file in the correct format

# ## Data fields
# - `document_text` - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
# - `question_text` - the question to be answered
# - `long_answer_candidates` - a JSON array containing all of the plausible long answers.
# - `annotations` - a JSON array containing all of the correct long + short answers. Only provided for train.
# - `document_url` - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
# - `example_id` - unique ID for the sample.

# ## Submission File
# Let's check the submission file to understand better what we need to predict
# For each ID in the test set, you must predict 
# 
# a) a set of start:end token indices, 
# 
# b) a YES/NO answer if applicable (short answers ONLY), or 
# 
# c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:
# 
# - -7853356005143141653_long,6:18
# - -7853356005143141653_short,YES
# - -545833482873225036_long,105:200
# - -545833482873225036_short,
# - -6998273848279890840_long,
# - -6998273848279890840_short,NO

# ## Load Data

# In[ ]:


path = '/kaggle/input/tensorflow2-question-answering/'
train_path = 'simplified-nq-train.jsonl'
test_path = 'simplified-nq-test.jsonl'
sample_submission_path = 'sample_submission.csv'


# In[ ]:


def read_data(path, sample = True, chunksize = 30000):
    if sample == True:
        df = []
        with open(path, 'rt') as reader:
            for i in range(chunksize):
                df.append(json.loads(reader.readline()))
        df = pd.DataFrame(df)
        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    else:
        df = pd.read_json(path, orient = 'records', lines = True)
        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        gc.collect()
    return df


# In[ ]:


train = read_data(path+train_path, sample = True)
test = read_data(path+test_path, sample = False)
train.head()


# In[ ]:


test.head()


# In[ ]:


sample_submission = pd.read_csv(path + sample_submission_path)
print('Our sample submission have {} rows'.format(sample_submission.shape[0]))
sample_submission.head()


# ## Missing Values

# In[ ]:


def missing_values(df):
    df = pd.DataFrame(df.isnull().sum()).reset_index()
    df.columns = ['features', 'n_missing_values']
    return df
missing_values(train)


# In[ ]:


missing_values(test)


# we don't have missing values.

# **Let's explore the first row of out train set to understand the logic of this dataset.**

# In[ ]:


question_text_0 = train.loc[0, 'question_text']
question_text_0


# In[ ]:


document_text_0 = train.loc[0, 'document_text'].split()
" ".join(document_text_0[:100])


# In[ ]:


from IPython.display import HTML,display
display(HTML(" ".join(document_text_0)))


# So in the `document_text` have a huge wikipedia text.

# In[ ]:


long_answer_candidates_0 = train.loc[0, 'long_answer_candidates']
long_answer_candidates_0[0:20]


# This are all the possibles long answers ranges. In other words they give you the start indices and last indices of all the possibles long answers in the document text columns that could answer the question.

# In[ ]:


annotations_0 = train.loc[0, 'annotations']
annotations_0


# ### More will be coming soon...

# In[ ]:




