#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# In[2]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[3]:


len(train_data)


# In[4]:


len(test_data)


# In[5]:


sns.heatmap(train_data.isna())


# In[6]:


sns.heatmap(test_data.isna())


# In[7]:


train_data.head()


# In[8]:


train_data.info()


# In[9]:


pd.options.display.max_colwidth=500
train_data.comment_text[0:5]


# In[10]:


train_data['text'] = train_data.comment_text.apply(lambda x: x.replace('\n', ' '))
test_data['text'] = test_data.comment_text.apply(lambda x: x.replace('\n', ' '))


# In[11]:


cats = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_prepared_data = []

def format_text_spacy(text):
    return (text.text, {'cats': {cat: text[cat] for cat in cats}})
    
for i in range(0,len(train_data)):
    text = train_data.iloc[i]
    train_prepared_data.append(format_text_spacy(text))


# In[12]:


train_prepared_data[0:3]


# In[13]:


import random
import spacy
import time
from spacy.util import minibatch, compounding

# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en')
textcat = nlp.create_pipe('textcat')
nlp.add_pipe(textcat, last=True)
for cat in cats:
    textcat.add_label(cat)


# In[14]:


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print("Training the model...")
    # Use more iters
    for i in range(1):
        start_time = time.time()
        losses = {}
        # batch up the examples using spaCy's minibatch
        # Use all data
        batches = minibatch(train_prepared_data[0:10000], size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        print(f"--- {time.time() - start_time} seconds ---")
            


# In[15]:


test = nlp("Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.")


# In[16]:


test.cats


# In[17]:


# submission = test_data[['id']]
# for cat in cats:
#     submission[cat] = 0

# for i in range(0,len(test_data)):
#     cats_probas = nlp(train_data.iloc[i].text).cats
#     for key in cats_probas.keys():
#         submission.loc[i,key] = cats_probas[key]


# In[18]:


# submission.info()


# In[19]:


# submission.head()


# In[20]:


# submission.to_csv('submission.csv', index=False)


# In[ ]:




