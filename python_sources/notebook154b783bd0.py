#!/usr/bin/env python
# coding: utf-8

# In[100]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scattertext as st
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from IPython.core.display import HTML
from IPython.display import IFrame
import IPython

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[6]:


df = pd.read_csv('../input/scrubbed.csv')


# In[7]:


df = df[df['comments'].apply(lambda x: type(x) == str and len(x) > 20)]


# In[8]:


df['parse'] = df.comments.apply(st.whitespace_nlp_with_sentences)


# In[28]:


len(df['parse']), sum(df['parse'].apply(lambda x: len(x.sents)) == 1)


# In[86]:


import re
nummer = re.compile('[0-9.]+')
df['duration (seconds)'] = df['duration (seconds)'].apply(lambda x: float(''.join(nummer.findall(str(x)))))
df['duration'] = df['duration (seconds)'].apply(lambda x: '>30 min' if x > 30 * 60 else '<=1 min' if x < 60 else 'middle')


# In[88]:


df.duration.value_counts()


# In[89]:


corpus = st.CorpusFromParsedDocuments(df, category_col='duration', parsed_col='parse').build()


# In[90]:


corpus = corpus.remove_categories(['middle'])


# In[91]:


compact_corpus = corpus.get_unigram_corpus().compact(st.ClassPercentageCompactor(st.OncePerDocFrequencyRanker, 2))


# In[93]:


meta_df = compact_corpus.get_df() 
html = st.produce_frequency_explorer(
    compact_corpus,
    category='>30 min',
    category_name='>30 Minutes',
    not_category_name='<=1 Minute',
    width_in_pixels=1000,
    use_full_doc=True,
    term_scorer=st.RankDifference(),
    term_ranker=st.OncePerDocFrequencyRanker,
    metadata=(meta_df['datetime'].apply(str) 
              + '; ' + meta_df['city'].apply(str) + ', ' + meta_df['state'].apply(str) 
              + '; ' + meta_df['duration (hours/min)'].apply(str))
)


# In[103]:


fn = 'long_vs_short_plot.html'
open(fn, 'w').write(html.replace('http://', 'https://'))
#IFrame(src=fn, width=1200, height=800)


# In[102]:


import IPython
url = 'https://www.kaggleusercontent.com/kf/2806061/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..AomAimaJmRrFHhPCsGn_-g.EjgqfXvNTSnrgn8xdUw8bMQqhWptZ9ubkaX0kCN-g-TVSiHffzKy4BBSLzvvSwktcg5aVkM1zmwkehQtIfs2Lvfe2dKt1J9xiMG3X8qUthQcZQj8s7QQqBMSW__murGv.nKR2KWctsEAKoGr23Kl_wQ/long_vs_short_plot.html'
iframe = '<iframe src=' + url + ' width=1200 height=800></iframe>'
IPython.display.HTML(iframe)

