#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will be analysing research papers on COVID-19. We will be using multiple NLP methods to analyse the data.
# 
# 1. Entity Extraction with Sci-Spacy
# 2. Keyphrases Extraction with TextRank
# 3. Keyphrases Extraction with TopicRank

# We won't be loading and working on all the data provided. We will using only papers published under **biorxiv_medrxiv**

# In[ ]:


import numpy as np 
import pandas as pd 

json_file_paths = []  # Intializing empty list to save full paths of json files to read them later on

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        a = os.path.join(dirname, filename)
        if a.endswith('.json') and 'biorxiv_medrxiv' in a:
            json_file_paths.append(a)
        elif a.endswith('.csv'):
            print(a)


# In[ ]:


# Number of json files
len(json_file_paths)


# Download and install ***pke for keyphrase extraction***
# 
# > *pke is an open source python-based keyphrase extraction toolkit. It provides an end-to-end keyphrase extraction pipeline in which each component can be easily modified or extended to develop new models. pke also allows for easy benchmarking of state-of-the-art keyphrase extraction models, and ships with supervised models trained on the SemEval-2010 dataset.*

# In[ ]:


get_ipython().system('pip install git+https://github.com/boudinfl/pke.git')
get_ipython().system('python -m nltk.downloader stopwords')
get_ipython().system('python -m nltk.downloader universal_tagset')
get_ipython().system('python -m spacy download en')


# Installing Sci spacy for medical entities extraction

# > *ScispaCy is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2). AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
# For more information, visit https://github.com/allenai/scispacy*

# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz')


# In[ ]:


import json, spacy, scispacy, en_ner_bc5cdr_md
import pandas as pd
from pke.unsupervised import TopicRank, TextRank, YAKE


# In[ ]:


def read_jsons(file_name):
    with open(file_name, 'r') as f:
        json_data = json.load(f)
    return json_data


# In[ ]:


data = pd.DataFrame([read_jsons(i) for i in  json_file_paths])


# In[ ]:


data = data.fillna(' ')


# In[ ]:


def process_abstract_body(value):
    if value != ' ':
        return " ".join(i['text'] for i in value)
    else:
        return ' '


# In[ ]:


def get_title(value):
    try:
        return value.get('title')
    except:
        return ' '


# In[ ]:


data.abstract = data.abstract.apply(process_abstract_body)


# In[ ]:


data.body_text = data.body_text.apply(process_abstract_body)


# In[ ]:


data["title"] = data.metadata.apply(get_title)


# In[ ]:


data = data[['paper_id', 'title', 'abstract', 'body_text']]


# In[ ]:


data.head()


# In[ ]:


def use_textrank(text, num_keyphrases=5):
    extractor = TextRank()
    extractor.load_document(text)
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(window=2, top_percent=10)
    keys = []
    scores = []
    for (keyphrase, score) in extractor.get_n_best(n=num_keyphrases):
        keys.append(keyphrase)
        scores.append(score)
    return keys, scores


# In[ ]:


for i,j in zip(data.body_text[:5],data.title[:5]):
    keys, scores = use_textrank(i)
    key_data = pd.DataFrame({"keyphrases":keys, "scores":scores})
    print("Title is "+j)
    print(key_data)


# In[ ]:


def use_topicrank(text, num_keyphrases=10):
    extractor = TopicRank()
    extractor.load_document(text)
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(threshold=0.7,
                                  method='average')
    keys = []
    scores = []
    for (keyphrase, score) in extractor.get_n_best(n=num_keyphrases):
        keys.append(keyphrase)
        scores.append(score)
    return keys, scores


# In[ ]:


for i,j in zip(data.body_text[:5],data.title[:5]):
    keys, scores = use_topicrank(i)
    key_data = pd.DataFrame({"keyphrases":keys, "scores":scores})
    print("Title is "+j)
    print(key_data)


# In[ ]:


nlp = en_ner_bc5cdr_md.load()


# In[ ]:


doc = nlp(data.body_text[0])
covid_ents = list(set([(i.text,i.label_) for i in doc.ents if i.label_ == 'DISEASE']))
covid_ents


# In[ ]:




