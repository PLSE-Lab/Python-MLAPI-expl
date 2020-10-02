#!/usr/bin/env python
# coding: utf-8

# Hello. 
# 
# In the process of studying the kernels of a given competitions, I often saw a code that is used for estimate cover by a dictionary of vector representations.
# 
# This method is good, but it looks "not pythonically" outwardly. For example, I do not understand the point of using "try-except". Below is my solution.

# In[ ]:


import pandas as pd
import numpy as np

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer

tqdm.pandas()

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


# In[ ]:


def build_vocab(texts):
    sentences = texts.progress_apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)
    
embedding_index = load_embeddings("../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec")


# In[ ]:


tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(train.comment_text.tolist() + test.comment_text.tolist())


# This is my solution (I hope.. I have not met a kernel with something similar)

# In[ ]:


get_ipython().run_cell_magic('time', '', "def check_coverage_new(word_counts, wanted_keys):\n    a = {key: val for key, val in word_counts.items() if key not in wanted_keys}\n    print(f'Found embeddings for {1-len(a)/len(word_counts):.2%} of vocablen')\n    print(f'Found embeddings for {1-sum(a.values())/sum(word_counts.values()):.2%} of all text')\n    return sorted(a.items(), key= lambda x : x[1], reverse=True)\n\nwanted_keys = embedding_index.keys()\nsort_x = check_coverage_new(tokenizer.word_counts, wanted_keys)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport operator \n\ndef check_coverage(vocab, embeddings_index):\n    a = {}\n    oov = {}\n    k = 0\n    i = 0\n    for word in tqdm(vocab):\n        try:\n            a[word] = embeddings_index[word]\n            k += vocab[word]\n        except:\n\n            oov[word] = vocab[word]\n            i += vocab[word]\n            pass\n\n    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))\n    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))\n    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]\n\n    return sorted_x\n\nsort_y = check_coverage(tokenizer.word_counts, embedding_index)")


# In the course of work, you will use the "check_coverage_new" function several times, but you no longer need to override the "wanted_keys" (if you use one set of vector views), which speeds up the definition.

# In[ ]:


get_ipython().run_cell_magic('time', '', "def check_coverage_new(word_counts, wanted_keys):\n    a = {key: val for key, val in word_counts.items() if key not in wanted_keys}\n    print(f'Found embeddings for {1-len(a)/len(word_counts):.2%} of vocablen')\n    print(f'Found embeddings for {1-sum(a.values())/sum(word_counts.values()):.2%} of all text')\n    return sorted(a.items(), key= lambda x : x[1], reverse=True)\n\nsort_x = check_coverage_new(tokenizer.word_counts, wanted_keys)")


# In[ ]:


sort_x[:10], sort_y[:10]


# Total:
# 1. Execution speed at ~ 100ms- ~ 200ms faster. On this dataset this is not critical, but in larger datasets this can be a significant plus;
# 2. The function has become shorter;
# 3. One less import.
# 
# I hope my decision will be useful for you.
