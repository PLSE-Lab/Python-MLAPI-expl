#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# Any results you write to the current directory are saved as output.


# In[ ]:


#!conda install -y faiss-gpu cudatoolkit=10.0 -c pytorch
#!pip install tensorflow_text
#!pip install tensorflow-gpu


# In[ ]:


import glob
import json
TITLE_DATA = []

for f in glob.glob("/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    TITLE_DATA.append((data['paper_id'], data['metadata']['title']))
    
for f in glob.glob("/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    TITLE_DATA.append((data['paper_id'], data['metadata']['title']))

for f in glob.glob("/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    TITLE_DATA.append((data['paper_id'], data['metadata']['title']))
    
for f in glob.glob("/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    TITLE_DATA.append((data['paper_id'], data['metadata']['title']))


# In[ ]:


SENTENCE_DATA = []
PARAGRAPH_ID = 0

for f in glob.glob("/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    for text in data['body_text']:
        PARAGRAPH_ID += 1
        for sent in text['text'].split("."):
            SENTENCE_DATA.append((data['paper_id'], PARAGRAPH_ID, sent, text['text']))
            
for f in glob.glob("/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/*.json"):
    data = json.loads(open(f, "r").read().strip())
    for text in data['body_text']:
        PARAGRAPH_ID += 1
        for sent in text['text'].split("."):
            SENTENCE_DATA.append((data['paper_id'], PARAGRAPH_ID, sent, text['text']))


# In[ ]:


df_title = pd.DataFrame(TITLE_DATA, columns=['paper_id', 'title'])
#df_sent = pd.DataFrame(SENTENCE_DATA, columns=['paper_id', 'paragraph_id', 'sentence', 'text'])
#df = df_sent.merge(df_title, how="left", on=["paper_id"])
#df['d_text'] = df['title'] + " " + df['sentence']
df = df_title.copy()
df['id'] = df.index


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
tf.executing_eagerly()


# In[ ]:


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


# In[ ]:


train = df.title.tolist()
list_train = [train[i:i+100] for i in range(0,len(train),100)]


# In[ ]:


import faiss

dimension = 512
nlist = 5  # number of clusters
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)


# In[ ]:


for x in list_train:
    db_vectors = embed(x).numpy()
    print(index.is_trained)   # False
    index.train(db_vectors)  # train on the database vectors
    print(index.ntotal)   # 0
    index.add(db_vectors)   # add the vectors and update the index
    print(index.is_trained)  # True
    print(index.ntotal)  


# In[ ]:


#!mkdir -p /kaggle/output

faiss.write_index(index,"/kaggle/output/index_v1")


# <div class='alert alert-success'>
# <h1> TASK-1) What is known about transmission, incubation, and environmental stability?</h1>
# </div>
# <div class='alert alert-warning'>
# <h2>
# (Task Details)
# What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# </h2>
# </div>

# ### Q) **Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.**

# In[ ]:


from  more_itertools import unique_everseen


inp_query = """Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery."""
query_vectors = embed([inp_query]).numpy()
answer = []
distances, indices = index.search(query_vectors, 100)
for ind in list(unique_everseen(indices[0])):
    title = df[df['id']==ind].title.tolist()[0]
    #context_text = df[df['id']==ind].text.tolist()[0]
    paper_id = df[df['id']==ind].paper_id.tolist()[0]
    answer.append((paper_id, title))

df_ans = pd.DataFrame(answer, columns=['Paper_id', 'Title'])
df_ans.head(10)


# <div class='alert alert-success'>
# <h1>REAL TIME QUESTION ANSWER</h1>
# </div>

# In[ ]:


from  more_itertools import unique_everseen
while True:
    print("Enter Query? ... Press 'q' to Quit")
    print("=="*10)
    inp_query = input()
    if inp_query in ['q', 'Q']:
        break
    query_vectors = embed([inp_query]).numpy()
    print("Answer")
    distances, indices = index.search(query_vectors, 50)
    for ind in list(unique_everseen(indices[0])):
        print("TITLE: ", df[df['id']==ind].title.tolist()[0])
        #print("Text: ", df[df['id']==ind].text.tolist()[0])
    print("=="*10)


# In[ ]:


import pandas as pd

df1 = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")


# In[ ]:


df1.columns


# In[ ]:


df1.info()


# In[ ]:


df1.url


# In[ ]:




