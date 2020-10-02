#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
from scipy.sparse import save_npz
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


start = time.time()
documents = [TaggedDocument(word_tokenize(doc.lower()), [i]) for i, doc in enumerate(train_df['question_text'].values)]
print(f'Completed in {time.time()-start:.2f} sec')


# In[ ]:


start = time.time()

model = Doc2Vec(
    documents=documents,
    vector_size=200,
    epochs=25,
    window=5,
    workers=4,
    dm=1
)

print(f'Completed in {time.time()-start:.2f} sec')
model.save("d2v.model")


# In[ ]:


docvecs = np.stack([model.docvecs[x] for x in range(train_df.shape[0])])
np.save('docvecs.npy', docvecs)

