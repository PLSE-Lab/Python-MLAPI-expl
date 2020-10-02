#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gensim
from gensim.corpora.textcorpus import TextCorpus
from scipy.special import softmax
import plotly.express as px


# In[ ]:


def build_isolator(chars):
    def isolate(text):
        for c in chars:
            text = text.replace(c, f" {c}")
        return text
    return isolate


# In[ ]:


raw = pd.read_csv('/kaggle/input/covid19-public-health-news-by-cdc-and-who/data.csv')
raw.content = (
    raw.content
    .astype(str)
    .str.replace(u'\xa0', u' ')
)


# In[ ]:


px.histogram(raw, x='type', color='source')


# ## Text preprocessing

# In[ ]:


isolate_punc = build_isolator('~!@#$%^&*()_+`-={}|:<>?[]/;\',."')


# In[ ]:


raw['processed'] = (
    raw.content
    .apply(isolate_punc)
    .str.lower()
)


# ## Build corpus

# In[ ]:


get_ipython().run_cell_magic('time', '', 'paragraphs = []\n\nfor i in range(raw.shape[0]):\n    r = raw.loc[i]\n    processed = r.processed.split("\\n")\n    original = r.content.split("\\n")\n    \n    for p, o in zip(processed, original):\n        paragraphs.append([p, o, r.url, r.date, r.source, r.type, r.title, r.content])\n\nnew_cols = [\'processed\', \'original\', \'url\', \'date\', \'source\', \'type\', \'title\', \'full_content\']\nparagraphs = pd.DataFrame(paragraphs, columns=new_cols)')


# ## Build search index

# In[ ]:


get_ipython().run_line_magic('time', 'corpus = [p.split() for p in paragraphs.processed]')
get_ipython().run_line_magic('time', 'bm25 = gensim.summarization.bm25.BM25(corpus)')


# ## Try querying

# In[ ]:


k = 10


# In[ ]:


input_text = "Hello world cat!"

query = isolate_punc(input_text.lower()).split()
query


# In[ ]:





# In[ ]:


get_ipython().run_line_magic('time', 'scores = softmax(bm25.get_scores(query))')
get_ipython().run_line_magic('time', 'top_matches = np.argsort(scores)[:-k-1:-1]')
get_ipython().run_line_magic('time', 'top_scores = scores[top_matches]')


# In[ ]:


retrieved = paragraphs.loc[top_matches].reset_index(drop=True)
retrieved['score'] = scores[top_matches]
retrieved.head()


# In[ ]:


def format_results(i, df):
    source = f"{df.source[i]} ({df.type[i]})" if str(df.type[i]) != 'nan' else df.source[i]
    out = f"""Result #{i+1}
> {df.original[i]}
Title: [{df.title[i]}]({df.url[i]})
Match score: {df.score[i] * 100:.2f}%
Source: {source}
    """
    
    return out


# In[ ]:


out = [format_results(i, retrieved) for i in retrieved.index]

print(("\n" + "="*30 + "\n\n").join(out))

