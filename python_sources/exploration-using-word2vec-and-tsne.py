#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Imports:


# In[ ]:


from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.summarization import keywords
from gensim.summarization import summarize


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')


# Set up logging

# In[ ]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#print(train_df.shape)
#print(test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


is_dup = train_df['is_duplicate'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(is_dup.index, is_dup.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Is Duplicate', fontsize=12)
plt.show()


# In[ ]:


is_dup / is_dup.sum()


# In[ ]:


all_question1 = train_df[[3]]
all_question2 = train_df[[4]]

all_ques_df = pd.DataFrame(pd.concat([train_df['question1'], train_df['question2']]))
all_ques_df.columns = ["questions"]

sentence1 = all_question1.question1[0]
sentence2 = all_question2.question2[0]

dummy_sentence = all_ques_df.questions[0]
print(dummy_sentence)


# In[ ]:


#convert into a list of words
#remove unnnecessary, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

print(sentence_to_wordlist(sentence1))


# In[ ]:


num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1


# In[ ]:


question2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[ ]:


question2vec.train(sentence1)


# In[ ]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[ ]:


question2vec.save(os.path.join("trained", "question2vec.w2v"))


# In[ ]:


question2vec = w2v.Word2Vec.load(os.path.join("trained", "question2vec.w2v"))


# In[ ]:


tsne = sklearn.manifold.TSNE(n_components=3, random_state=0)
all_word_vectors_matrix = question2vec.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1], coords[2])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[question2vec.wv.vocab[word].index])
            for word in question2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y", "z"]
)
points.head(10)


# In[ ]:




