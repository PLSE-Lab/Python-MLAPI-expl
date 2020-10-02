#!/usr/bin/env python
# coding: utf-8

# #### Transfer Learning in NLP -ULMFiT
# Authors: Vikas Kumar (vikkumar@deloitte.com) | Abhishek Aditya Kashyap (abhikashyap@deloitte.com)

# **References:**
# * https://github.com/fastai/fastai/blob/master/examples/ULMFit.ipynb
# * https://github.com/fastai/course-nlp/blob/master/5-nn-imdb.ipynb
# * http://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html

# ### Language Modeling & Sentiment Analysis of IMDB movie reviews

# We will be looking at IMDB movie reviews.  We want to determine if a review is negative or positive, based on the text.  In order to do this, we will be using **transfer learning**.
# 
# Transfer learning has been widely used with great success in computer vision for several years, but only in the last year or so has it been successfully applied to NLP (beginning with ULMFit, which we will use here, which was built upon by BERT and GPT-2).
# 
# As Sebastian Ruder wrote in [The Gradient](https://thegradient.pub/) last summer, [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/output'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# importing fastai libraries

# In[ ]:


sample=False


# In[ ]:


from fastai import *
from fastai.text import *


# **SELECTING DEVICE: CPU/CUDA**

# In[ ]:


defaults.device = torch.device('cuda',0) if torch.cuda.is_available() else torch. device('cpu')
defaults.device


# In[ ]:


DATA_PATH = Path('/kaggle/input/usinlppracticum/')
DATA_PATH.ls()


# #### data Preparation for Language Model data

# In[ ]:


lm_data=pd.read_csv(DATA_PATH/'imdb_train.csv')
lm_data.head()


# In[ ]:


lm_data1=pd.read_csv(DATA_PATH/'imdb_train.csv')
lm_data1['sentiment']=0
lm_data2=pd.read_csv(DATA_PATH/'imdb_test.csv')
lm_data2['sentiment']=0
lm_data= pd.concat([lm_data1, lm_data2], ignore_index=True)
lm_data=lm_data[['review','sentiment']]
lm_data.to_csv('lm_data.csv',index=False)
lm_data.shape


# In[ ]:


if sample:
    lm_data=pd.read_csv('lm_data.csv').sample(10000).reset_index(drop=True)
else:
    lm_data=pd.read_csv('lm_data.csv')
#------------
lm_data.head()


# ###  splitting langauge model data

# In[ ]:


from sklearn.model_selection import train_test_split

train_lm, val_lm = train_test_split(lm_data,test_size=0.10)
train_lm.shape,val_lm.shape


# ### Creating the TextLMDataBunch
# This is where the unlabelled data is going to be useful to us, as we can use it to fine-tune our model. Let's create our data object with the data block API (next line takes a few minutes).

#  We first have to convert words to numbers. This is done in two differents steps: 
# *  tokenization 
# * numericalization. 
# 
# A `TextDataBunch` does all of that behind the scenes for you.

# In[ ]:


data_lm = TextLMDataBunch.from_df(DATA_PATH, train_lm,val_lm,text_cols='review', label_cols='sentiment')
data_lm.save('/kaggle/working/data_lm_export.pkl')


# ### Tokenization
# The first step of processing we make texts go through is to split the raw sentences into words, or more exactly tokens. The easiest way to do this would be to split the string on spaces, but we can be smarter:
# 
# - we need to take care of punctuation
# - some words are contractions of two different words, like isn't or don't
# - we may need to clean some parts of our texts, if there's HTML code for instance
# 
# To see what the tokenizer had done behind the scenes, let's have a look at a few texts in a batch.
# 
# The texts are truncated at 100 tokens for more readability. We can see that it did more than just split on space and punctuation symbols: 
# - the "'s" are grouped together in one token
# - the contractions are separated like his: "did", "n't"
# - content has been cleaned for any HTML symbol and lower cased
# - there are several special tokens (all those that begin by xx), to replace unkown tokens (see below) or to introduce different text fields (here we only have one).

# ### Numericalization
# Once we have extracted tokens from our texts, we convert to integers by creating a list of all the words used. We only keep the ones that appear at list twice with a maximum vocabulary size of 60,000 (by default) and replace the ones that don't make the cut by the unknown token `UNK`.
# 
# The correspondance from ids tokens is stored in the `vocab` attribute of our datasets, in a dictionary called `itos` (for int to string).

# In[ ]:


data_lm.vocab.itos[:10]


# #### And if we look at what a what's in our datasets, we'll see the tokenized text as a representation:

# In[ ]:


data_lm.train_ds[0][0]


# ### But the underlying data is all numbers

# In[ ]:


data_lm.train_ds[0][0].data[:100]


# In[ ]:


len(data_lm.vocab.itos),len(data_lm.train_ds)


# In[ ]:


data_lm.train_ds[0][0].data.shape


# In[ ]:


data_lm.show_batch()


# In[ ]:


learn_lm = language_model_learner(data_lm, AWD_LSTM)


# ### loading wikitext vocab

# In[ ]:


import pickle
wiki_itos = pickle.load(open('/kaggle/input/wiki-vocab/itos_wt103.pkl', 'rb'))


# In[ ]:


wiki_itos[:10]


# In[ ]:


vocab = data_lm.vocab


# In[ ]:


vocab.stoi["stingray"]


# In[ ]:


vocab.itos[vocab.stoi["stingray"]]


# In[ ]:


vocab.itos[vocab.stoi["mobula"]]


# In[ ]:


awd = learn_lm.model[0]
print(awd)


# In[ ]:


enc = learn_lm.model[0].encoder


# In[ ]:


enc.weight.size()


# ### Difference in vocabulary between IMDB and Wikipedia
# We will compare the `vocabulary from wikitext with the vocabulary in IMDB`.  It is to be expected that the two sets have some different vocabulary words, and that is no problem for `transfer learning!`

# In[ ]:


len(wiki_itos)


# In[ ]:


len(vocab.itos)


# In[ ]:


i, unks = 0, []
while len(unks) < 50:
    if data_lm.vocab.itos[i] not in wiki_itos: unks.append((i,data_lm.vocab.itos[i]))
    i += 1


# In[ ]:


wiki_words = set(wiki_itos)
imdb_words = set(vocab.itos)


# In[ ]:


wiki_not_imbdb = wiki_words.difference(imdb_words)
imdb_not_wiki = imdb_words.difference(wiki_words)


# In[ ]:


wiki_not_imdb_list = []

for i in range(100):
    word = wiki_not_imbdb.pop()
    wiki_not_imdb_list.append(word)
    wiki_not_imbdb.add(word)


# In[ ]:


wiki_not_imdb_list[:15]


# In[ ]:


imdb_not_wiki_list = []

for i in range(100):
    word = imdb_not_wiki.pop()
    imdb_not_wiki_list.append(word)
    imdb_not_wiki.add(word)


# In[ ]:


imdb_not_wiki_list[:15]


# ### All words that appear in the IMDB vocab, but not the wikitext-103 vocab, will be initialized to the same random vector in a model.  `As the model trains, we will learn these weights.`

# ### Generating fake movie reviews (using wiki-text model)

# In[ ]:


TEXT = "The color of the sky is"
N_WORDS = 40
N_SENTENCES = 2

print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[ ]:


# doc(LanguageLearner.predict)


# In[ ]:


print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.20) for _ in range(N_SENTENCES)))


# ### Training the Langauge Model

# In[ ]:


learn_lm.fit_one_cycle(1, 2e-2, moms=(0.8,0.7), wd=0.1)


# In[ ]:


learn_lm.unfreeze()
learn_lm.fit_one_cycle(10, 2e-3, moms=(0.8,0.7), wd=0.1)


# In[ ]:


learn_lm.path = Path('/kaggle/working') 
learn_lm.model_dir= Path('.')


# In[ ]:


learn_lm.save_encoder('fine_tuned_enc')


# ### More generated movie reviews
# How good is our model? Well let's try to see what it predicts after a few given words.

# In[ ]:


TEXT = "i liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# ### Training Classifier on finetuned Language Models

# In[ ]:


if sample:
    data_cls=pd.read_csv(DATA_PATH/'imdb_train.csv').sample(1000).reset_index(drop=True)
else:
    data_cls=pd.read_csv(DATA_PATH/'imdb_train.csv')
#----------
data_cls.head()


# In[ ]:


# Classifier model data
from sklearn.model_selection import train_test_split
train, val = train_test_split(data_cls,test_size=0.10, random_state=42)
label_col= 'sentiment'

label_mapping= {'negative':0,'positive':1}
train[label_col]=train[label_col].map(label_mapping)
val[label_col]=val[label_col].map(label_mapping)
train.head()


# In[ ]:


data_clas = TextDataBunch.from_df(DATA_PATH, train, val,
                  vocab=data_lm.train_ds.vocab,
                  text_cols="review",
                  label_cols='sentiment',
                  bs=64,device = defaults.device)


# In[ ]:


learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3) #.to_fp16()
learn_c.path = Path('/kaggle/working') 
learn_c.model_dir= Path('.')
learn_c.load_encoder('fine_tuned_enc')
learn_c.freeze()


# In[ ]:


learn_c.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[ ]:


learn_c.freeze_to(-2)
learn_c.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn_c.freeze_to(-3)
learn_c.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn_c.unfreeze()
learn_c.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

