#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 Biobert Semantic Search Engine**
# 1. **The goal is to find semantically similar articles based on title by mapping variable length sentences to a fixed length vector using Biobert model.** 
# 2. **The intention is to implement this using bert as a service. **
# 3. **Also abstracts and full text can be vectorized using this. (This kernel uses only biorxiv dataset and is implemented for titles only)**
# 4. **This is also worth trying with ClinicalBert, SciBert and other Bert based biomedical language models.**
# 
# **The next goal would be trying to create an end to end pipeline having a combination of different algorithms to have specific answers to the questions mentioned in the tasks.**
# 

# Installing the required dependencies.

# In[ ]:


get_ipython().system('pip uninstall tensorflow==2.1.0 --yes')
get_ipython().system('pip install bert-serving-server')
get_ipython().system('pip install bert-serving-client')
get_ipython().system('pip install --upgrade ipykernel')
get_ipython().system('pip install tensorflow==1.13.1')


# Downloading the pre-trained weights of Biobert

# In[ ]:


get_ipython().system('wget https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz')


# Unpacking the weights and renaming it as per the original bert model (as its a requirement for bert as a service)

# In[ ]:


get_ipython().system('tar xvzf biobert_v1.1_pubmed.tar.gz')
get_ipython().run_line_magic('cd', 'biobert_v1.1_pubmed')
get_ipython().system("rename 's/model.ckpt-1000000.data-00000-of-00001/bert_model.ckpt.data-00000-of-00001/' *")
get_ipython().system("rename 's/model.ckpt-1000000.meta/bert_model.ckpt.meta/' *")
get_ipython().system("rename 's/model.ckpt-1000000.index/bert_model.ckpt.index/' *")
get_ipython().system('ls #/kaggle/working/biobert_v1.1_pubmed')
#!port_num=5555


# Making Sure we have the right version of tensorflow installed

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# Importing the required packages

# In[ ]:


import pandas as pd
from bert_serving.client import BertClient
import numpy as np
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


# Setting required arguments for the bert server and starting the server.

# In[ ]:


a = get_args_parser().parse_args(['-model_dir', '/kaggle/working/biobert_v1.1_pubmed',
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu',
                                     '-num_worker','4'])
server = BertServer(a)
server.start()


# In[ ]:


bc = BertClient(port=5555, port_out=5556)


# Importing the biorxiv csv file and creating a list of all the titles

# In[ ]:


biorx_df = pd.read_csv('../../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')
biorx_lst = biorx_df['title'].astype(str).to_list()
print(biorx_lst)


# Creating embeddings of all the titles

# In[ ]:


doc_vecs = bc.encode(biorx_lst)
print(doc_vecs.shape)


# Function to find top K similar articles for a given query

# In[ ]:


def find_similar_articles(query,topk):
    query_vec = bc.encode([query])[0]
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], biorx_lst[idx]))


# **smoking or pre-existing pulmonary disease increase risk of COVID-19**

# In[ ]:


find_similar_articles("smoking or pre-existing pulmonary disease increase risk of COVID-19",5)


# **Socio-economic and behavioral factors to understand the economic impact of the coronavirus and whether there were differences.**

# In[ ]:


find_similar_articles("Socio-economic and behavioral factors to understand the economic impact of the coronavirus and whether there were differences.",5)


# Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

# In[ ]:


find_similar_articles("Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities",5)


# **Risk of of COVID-19 for neonates and pregnant women**

# In[ ]:


find_similar_articles("Risk of of COVID-19 for neonates and pregnant women",5)


# **Potential risk factors of Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors**

# In[ ]:


find_similar_articles(" Potential risk factors of Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",5)


# **What is the severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups?**

# In[ ]:


find_similar_articles("What is the severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups?",5)


# **Susceptibility of populations**

# In[ ]:


find_similar_articles("Susceptibility of populations",5)


# **Public health mitigation measures that could be effective for control**

# In[ ]:


find_similar_articles("Public health mitigation measures that could be effective for control",5)


# In[ ]:


get_ipython().system('bert-serving-terminate -port 5555')


# **References:**
# 
# **Bert as a service: https://github.com/hanxiao/bert-as-service**
# 
# **Biobert pre-trained weights: https://github.com/naver/biobert-pretrained/releases**
# 
# **Dataset Source: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv**
# 
