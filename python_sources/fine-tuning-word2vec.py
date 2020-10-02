#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
import pandas as pd
from nltk.tokenize import RegexpTokenizer

forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")


# In[ ]:


# take first 100 forum posts
sentences = forum_posts.Message.astype('str').tolist()

# toeknize
tokenizer = RegexpTokenizer(r'\w+')
sentences_tokenized = [w.lower() for w in sentences]
sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]


# In[ ]:


model = KeyedVectors.load_word2vec_format("../input/word2vec-google/GoogleNews-vectors-negative300.bin",
                                         binary = True)


# Todo: figure out how to update the loaded word vectors
# 
# SO question w/ more info https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights

# In[ ]:


model_2 = Word2Vec(size=300, min_count=1)
model_2.build_vocab(sentences_tokenized)
total_examples = model_2.corpus_count
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("../input/word2vec-google/GoogleNews-vectors-negative300.bin", binary=True)
model_2.train(sentences_tokenized, total_examples=total_examples, epochs=model_2.iter)


# In[ ]:


# original model does not have Kaggle in it
model.similarity('kaggle','google')


# In[ ]:


# tuned model does, and it's pretty close to Google! :3
model_2.similarity('kaggle','google')

