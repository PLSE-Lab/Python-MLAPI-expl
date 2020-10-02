#!/usr/bin/env python
# coding: utf-8

# <h1> COVID-19 - Search Engine with Bert Score </h1>
# 
# ![](https://cdn.drawception.com/drawings/661287/54Z1tooqg4.png)
# 
# This notebook is inspired by CORD-19 Solution Toolbox released by Gabriel Preda. Thanks for sharing!
# 
# The goal is really simple! We want to find the articles who talk about a specific topics. The idea of this notebook is to provide to the community the possiiblity to put in input a question, a topic (defined by keywords or sentences) or just keywords and use BERT to vectorize the input, vectorize the abstract of the articles and compare the similarity between them. Like that you can identify one or more articles who are a strong similarity (With cosinus similarity).
# 
# # 1. Import & install the packages needed
# * 
# We have to install a package in order use BERT but for sentences. For that I use the package sentence transformers. Here you can find more informations: https://github.com/UKPLab/sentence-transformers. I am not here to explain BERT or Cosinu Similarity. You can find a lot of information about these topics. I want to stay pragmatic and give the possibility to re-use this notebook for other project.

# In[ ]:


get_ipython().system('pip install -U sentence-transformers')


# In[ ]:


import numpy as np
import pandas as pd
import scipy as sc

import os
import json
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

t = time.time()
elapsed = time.time() - t

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))
        
warnings.filterwarnings("ignore")
model = SentenceTransformer('bert-base-nli-mean-tokens')


# # Input - Questions or keywords
# In this part of the notebook, you have to put the question or the keywords related to the topic that you want to identify through the research papers. Example with the second task of the challenge.

# In[ ]:


question_embedding = model.encode(['What do we know about virus genetics, origin, and evolution?'])

queries = ['What is known about transmission, incubation, and environmental stability?', 'What do we know about COVID-19 risk factors?', 
           'What do we know about virus genetics, origin, and evolution?', 'What do we know about vaccines and therapeutics?',
           'Are there geographic variations in the rate of COVID-19 spread?', 'Are there geographic variations in the mortality rate of COVID-19?',
           'Is there any evidence to suggest geographic based virus mutations?','What do we know about diagnostics and surveillance?',
           'What do we know about non-pharmaceutical interventions?','What has been published about medical care?',
           'What has been published about ethical and social science considerations?', 'What has been published about information sharing and inter-sectoral collaboration?']

query_embeddings = model.encode(queries)


# # Get the data
# The part about the collect was developped by Gabriel Preda. Thanks for sharing again :)! There too much data in this challenge. In order to show you how to use the search engine, I will use it just in a subset of the data in the folder comm_use_subset.

# In[ ]:


count = 0
file_exts = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        count += 1
        file_ext = filename.split(".")[-1]
        file_exts.append(file_ext)

file_ext_set = set(file_exts)
file_ext_list = list(file_ext_set)

count = 0
for root, folders, filenames in os.walk('/kaggle/input'):
    print(root, folders)
    
json_folder_path = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json"
json_file_name = os.listdir(json_folder_path)[0]
json_path = os.path.join(json_folder_path, json_file_name)

with open(json_path) as json_file:
    json_data = json.load(json_file)
    
json_data_df = pd.io.json.json_normalize(json_data)


# I get just 1000 research papers, but in this folder you can find more than 9k research papers. Of course, it is too long to get all of them through the notebook.

# In[ ]:


from tqdm import tqdm

# to process all files, uncomment the next line and comment the line below
#list_of_files = list(os.listdir(json_folder_path))
list_of_files = list(os.listdir(json_folder_path))[0:100]
comm_use_subset_df = pd.DataFrame()

for file in tqdm(list_of_files):
    json_path = os.path.join(json_folder_path, file)
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    json_data_df = pd.io.json.json_normalize(json_data)
    comm_use_subset_df = comm_use_subset_df.append(json_data_df)


# # Parser - Cleaning
# In order to use the text data, we have to parse it and clean it.

# In[ ]:


comm_use_subset_df


# In[ ]:


comm_use_subset_df['abstract_text'] = comm_use_subset_df['abstract'].apply(lambda x: x[0]['text'] if x else "")
comm_use_subset_df['abstract_text_cleaned'] = comm_use_subset_df['abstract_text'].str.replace('\d+', 'XXX')
comm_use_subset_df.reset_index(drop = True, inplace = True)


# # Bert Embedding
# Bert Embedding of the text data cleaned.
# 
# ![](https://cdn.drawception.com/drawings/915760/b8Taco5fmM.png)

# In[ ]:


with Timer('abstract_cleaned embeddings'):   
    abstract_embeddings = model.encode(comm_use_subset_df['abstract_text_cleaned'])


# # Bert Score for several questions

# In[ ]:


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = sc.spatial.distance.cdist([query_embedding], abstract_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(comm_use_subset_df['abstract_cleaned'][idx].strip(), "\n(Score: %.4f)" % (1-distance),"\n")


# # Bert Score for a specific question
# The Bert Score computed is the cosinus similarity between the Bert Embedding of the input and the abstract text cleaned.

# In[ ]:


question_abstract = []
bert_scores = []

for abstract_embedding, abstract_text in zip(abstract_embeddings, comm_use_subset_df['abstract_text']):
    bert_score = cosine_similarity([question_embedding[0], abstract_embedding])[1][0]
    question_abstract.append(bert_score)


# # Results

# In[ ]:


print("Index of the document: ", question_abstract.index(max(question_abstract)), "\nAbstract of the document: ", comm_use_subset_df['abstract_text'].ix[question_abstract.index(max(question_abstract))],
     "\nBert Similarity between the question and the document: ", max(question_abstract))

