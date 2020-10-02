#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# ### What is the goal of Project [COVIEWED](https://www.coviewed.org/)?
# 
# Project [COVIEWED](https://www.coviewed.org/)'s aim is to fight against misinformation on the web regarding the recent coronavirus pandemic / covid-19 outbreak. 
# 
# To achieve this, we collect different types of claims from web sources with supporting or attacking evidence. 
# 
# This information is used to train a machine learning classifier. 
# 
# Once trained, we plan to release a browser extension that highlights potential true/false claims on a web page to assist users in their information gathering process.
# 
# ### Organization
# 
# We have a public [Trello Board](https://trello.com/invite/b/jk00CW3u/9985740815b585156aaa22978a3067df/project-coviewed) and our project is completely open source and available on [Github](https://github.com/COVIEWED).

# ---

# In[ ]:


get_ipython().system('pip install -U stanza')


# In[ ]:


import os
import json
import stanza
import hashlib
import numpy as np
import pandas as pd


# In[ ]:


stanza.download('en')


# In[ ]:


nlp = stanza.Pipeline(processors='tokenize', lang='en', use_gpu=True)


# # Load a News Article

# In[ ]:


get_ipython().system('git clone https://github.com/COVIEWED/coviewed_web_scraping')


# In[ ]:


get_ipython().system('pip install -r coviewed_web_scraping/requirements.txt')


# In[ ]:


EXAMPLE_URL = 'https://www.euronews.com/2020/04/01/the-best-way-prevent-future-pandemics-like-coronavirus-stop-eating-meat-and-go-vegan-view'
print(EXAMPLE_URL)


# In[ ]:


get_ipython().system('cd coviewed_web_scraping/ && python3 src/scrape.py -u={EXAMPLE_URL}')


# In[ ]:


data_path = 'coviewed_web_scraping/data/'
fname = [f for f in os.listdir(data_path) if f.endswith('.txt')][0]
print(fname)
with open(os.path.join(data_path, fname), 'r') as my_file:
    txt_data = my_file.readlines()
txt_data = [line.strip() for line in txt_data if line.strip()]
len(txt_data)


# In[ ]:


article_url = txt_data[0]
print(article_url)
article_published_datetime = txt_data[1]
print(article_published_datetime)


# In[ ]:


article_title = txt_data[2]
print(article_title)


# In[ ]:


article_text = "\n\n".join(txt_data[3:])
print(article_text)


# # Sentence Segmentation

# In[ ]:


ALL_SENTENCES = []
txt = [p.strip() for p in article_text.split('\n') if p.strip()]
file_id = fname.split('.')[0]
print(file_id)
print()
for i, paragraph in enumerate(txt):
    doc = nlp(paragraph)
    for sent in doc.sentences:
        S = ' '.join([w.text for w in sent.words])
        sH = hashlib.md5(S.encode('utf-8')).hexdigest()
        print(sH)
        print(S)
        print()
        ALL_SENTENCES.append([file_id, sH, S])


# In[ ]:


fname = file_id+'_sentences.tsv'
print(fname)
AS = pd.DataFrame(ALL_SENTENCES, columns=['file_id','sentenceHash','sentence'])
len(AS)


# In[ ]:


AS.sample(n=min(len(AS),3))


# In[ ]:


AS.to_csv(fname, sep='\t', index=False, index_label=False)


# In[ ]:




