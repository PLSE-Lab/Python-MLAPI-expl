#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Based on [this](http://www.sciencemag.org/news/2020/03/who-launches-global-megatrial-four-most-promising-coronavirus-treatments) and some other resources, we know that there is some treatment types that has been using for coronavirus disease. I try to extract those traetments from datasets. All resources can be found below.

# ### method
# I basically read the datasets, search for exact words and make a worldcloud plot. 

# In[ ]:


from glob import glob
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os


# In[ ]:


dir_list = [
    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',
    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',
    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'
]
results_list = list()
for target_dir in dir_list:
    
    print(target_dir)
    
    for json_fp in glob(target_dir + '/*.json'):

        with open(json_fp) as json_file:
            target_json = json.load(json_file)

        data_dict = dict()
        data_dict['doc_id'] = target_json['paper_id']
        data_dict['title'] = target_json['metadata']['title']

        abstract_section = str()
        for element in target_json['abstract']:
            abstract_section += element['text'] + ' '
        data_dict['abstract'] = abstract_section

        full_text_section = str()
        for element in target_json['body_text']:
            full_text_section += element['text'] + ' '
        data_dict['full_text'] = full_text_section
        
        results_list.append(data_dict)
        
    
df_results = pd.DataFrame(results_list)
df_results.head()        


# In[ ]:


articles=df_results['full_text'].str.lower().values


# ## Remdesivir

# In[ ]:


Remdesivir=[]
for text in articles:
    for sentences in text.split('.'):
        if 'remdesivir' in sentences:
            Remdesivir.append(sentences)          


# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Remdesivir))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Chloroquine

# In[ ]:


Chloroquine=[]
for text in articles:
    for sentences in text.split('.'):
        if 'chloroquine' in sentences:
            Chloroquine.append(sentences)         


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Chloroquine))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## hydroxychloroquine

# In[ ]:


hydroxychloroquine=[]
for text in articles:
    for sentences in text.split('.'):
        if 'hydroxychloroquine' in sentences:
            hydroxychloroquine.append(sentences)            


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(hydroxychloroquine))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## lopinavir

# In[ ]:


lopinavir=[]
for text in articles:
    for sentences in text.split('.'):
        if 'lopinavir' in sentences:
            lopinavir.append(sentences)           


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(lopinavir))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Ritonavir

# In[ ]:


Ritonavir=[]
for text in articles:
    for sentences in text.split('.'):
        if 'ritonavir' in sentences:
            Ritonavir.append(sentences)          


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Ritonavir))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## interferon_beta

# In[ ]:


interferon_beta=[]
for text in articles:
    for sentences in text.split('.'):
        if 'interferon-beta' in sentences:
            interferon_beta.append(sentences) 


# In[ ]:


wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(interferon_beta))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Resources:

# * https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
# * https://www.kaggle.com/iancornish/drug-data
# * https://www.youtube.com/watch?v=S6GVXk6kbcs&lc=z23czv4rezzqspkcnacdp434abyko0xfj3zyelkza01w03c010c
# * https://www.kaggle.com/bgoss541/training-set-labeling-jump-start-umls-linking
# * https://www.kaggle.com/danofer/usp-drug-classification
# * https://stackoverflow.com/q/37470966
# * https://www.sciencemag.org/news/2020/03/who-launches-global-megatrial-four-most-promising-coronavirus-treatments#
# * https://stackoverflow.com/a/50737722
