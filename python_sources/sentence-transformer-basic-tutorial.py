#!/usr/bin/env python
# coding: utf-8

# # Using Sentence Transformer

# In[ ]:


get_ipython().system('pip install sentence-transformers')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


model = sentence_transformers.SentenceTransformer("../input/coronawhy/sentence_transformer_nli/model_results")


# In[ ]:


sents = model.encode(["coronavirus transmission mechanism", "SARs camel to human transmission"])
cosine_similarity(np.expand_dims(sents[0], axis=0), np.expand_dims(sents[1], axis=0))


# In[ ]:




