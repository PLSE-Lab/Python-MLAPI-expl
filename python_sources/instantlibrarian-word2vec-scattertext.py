#!/usr/bin/env python
# coding: utf-8

# # 1. Objective
# ### InstantLibrarian is an enhanced approach to search which applies Word2Vec, TF-IDF, Scattertext, and Cosine Similarity to deliver highly relevant search results with minimal words in the search query.
# 
# 
# 
# 

# # 2. Approach
# ### Applying unsepervised techniques will not ensure the relevance of search/cluser results:
# #### We apply Word2Vec (Word Embeddings) to explore/find out the relevant medical vocabulary/terms, the purpose is to use the similar/relevant words in a semi-supervised clustering.
# 

# # 3. Method
# ### The data used is a clean csv. version of the bioRxiv collection, papers with complete (non NaN) abstracts only were kept, as the text exploration with Scattertext will be done on abstracts only. We get a quick look at the data: 

# In[ ]:



import os

import numpy as np
import pandas as pd
from os import walk
for (dirpath, dirnames, filenames) in walk("../input"):
    print("Directory path: ", dirpath)
    print("Folder name: ", dirnames)
    print("File name: ", filenames)


# In[ ]:


data = pd.read_csv("../input/df-cov/df_COV.csv")
data.head()


# **The Above dataset is the merge result of bioRxiv json documents and the metadata file on paper id or sha, an extra column of publication year and month was added **

# # 4. Pipeline
# 
# ### a. Word2Vec 
# ### b. Categorize
# ### c. TF-IDF, Gensim using Scattertext 
# ### d. Recommender with Cosine Similarities 

# ## a. Word2Vec for Medical Terminology Exploration & Clustering
# ### In order to gain a good understanding of the different possible expressions in the medical literature which are relevent to the query/task questions, we first train a word2vec model on the corpus to learn more about all the medical vocabulary.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import scattertext as st #the main library used for corpous exploration
from pprint import pprint
import pandas as pd
import numpy as np
from scipy.stats import rankdata, hmean, norm
import spacy
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer
from collections import OrderedDict
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.matcher import Matcher
from gensim.models import word2vec
import re, io, itertools
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML

nlp = spacy.load('en_core_web_sm')
display(HTML("<style>.container { width:98% !important; }</style>"))


# In[ ]:


data = data.dropna(subset=['abstract_x'])


# In[ ]:


data = data.loc[data['year'] >= 2019]


# In[ ]:


data['parsed_text'] = data.text.apply(nlp)


# In[ ]:


data.head()


# In[ ]:


corpus = (st.CorpusFromParsedDocuments(data, category_col='Category', parsed_col='parsed_text')
          .build()
          .get_unigram_corpus())


# In[ ]:


model = word2vec.Word2Vec(size=100, window=5, min_count=10, workers=4)
model = st.Word2VecFromParsedCorpus(corpus, model).train(epochs=10000)


# ****

# ### [Answering the questions using a Word2Vec Model](https://fast-spire-21519.herokuapp.com/)
# #### An application of a simple Word2Vec model, built on the corpus of medical papers, feel free to explore using the hyperlink in the title above.

# ## What is the best method to combat the hypercoagulable state seen in COVID-19?
# 

# In[ ]:


model.wv.most_similar('hypercoagulable'),model.wv.most_similar('hypercoagulability'), model.wv.most_similar('clots')


# In[ ]:


hypercoagulable = ['hypercoagulable','hyperinflammatory', 'hypercoagulability', 'clots','coagulopathy', 'microcirculation','hypertrophy', 'vasoconstriction','stasis', 'vessel',  'transfusions']


# ## What is the efficacy of novel therapeutics being tested currently?
# 
# 

# In[ ]:


model.wv.most_similar('efficacy'), model.wv.most_similar('therapeutics'), model.wv.most_similar('treatment'), model.wv.most_similar('inhibitor')


# In[ ]:


therapeutic = ['immunogenicity','potency', 'potent','pharmacokinetics', 'antivirals', 'therapies','vaccines', 'drugs','therapeutic','repurposing','treatments','immunotherapy','adjuvants','countermeasures','prophylaxis','cure','oseltamivir','inhibitors','analog', 'protease','camostat','pikfyve', 'mesylate', 'rapamycin', 'adenosine' ]


# ### b. Categorization
# #### Using the keywords collected from the previous step (word2vec), we create more specific/relevant categories

# In[ ]:


temp=data.text.fillna("0")

data['Category'] = pd.np.where(temp.str.contains('|'.join(therapeutic)), "therapeutic",
                       pd.np.where(temp.str.contains('|'.join(hypercoagulable)), "hypercoagulable","other"))

data['Category'].value_counts()


# In[ ]:


corpus = st.CorpusFromParsedDocuments(data, category_col='Category', parsed_col='parsed_text').build()


# ### [c. TF-IDF, Gensim using Scattertext](https://emazika.github.io/Scatter-html/)
# #### Using Scattertext, to explore first hand the content of the corpus in terms of the relevant questions regarding therapeutics and hypercoagulable state. Feel free to explore using the hyperlink in the title.

# In[ ]:


target_term = 'coagulation'

html = st.word_similarity_explorer_gensim(corpus,
                                          category='therapeutic',
                                          category_name='therapeutic',
                                          not_category_name='hypercoagulable',
                                          target_term=target_term,
                                          minimum_term_frequency=200,
                                          width_in_pixels=1000,
                                          word2vec=model,
                                          metadata=data['title_x'])
file_name = 'COVID19_DEMO_similarity_gensim.html'
#open(file_name, 'wb').write(html.encode('utf-8'))
#IFrame(src=file_name, width = 1200, height=700)


# ### [d. Recommender](https://agile-basin-38883.herokuapp.com/)
# #### Using a simple Cosine Similarity model, a recommender is used to further mine all relevant articles to the specific questions above (about therapeutics or hypercoagulable state), feel free to explore using the hyperlink in the title.
# 
# 

# # 5. Results
# #### A total of 94 relevant research papers were collected using the steps explained above, the results will be updated continuously and the output data table will be improved to match the required formatting.

# In[ ]:


Final_submission = pd.read_csv("../input/final-submission/final_doc.csv")
Final_submission

