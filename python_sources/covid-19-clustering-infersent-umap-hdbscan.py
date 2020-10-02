#!/usr/bin/env python
# coding: utf-8

# # COVID-19 - A clustering approach (InferSent semantics embedding + Umap dimension reductioning + HDBScan clustering)
# ## 1. Semantics understanding
# 
# * InferSent semantics embedding (4096 features) - this is chosen according to http://hunterheidenreich.com/blog/comparing-sentence-embeddings/ as the top scorer model for sentence semantics relatedness (SICK-R)
# * Conversion of sentence embedding to paragraph is simply the average of the vectors (Iyyer et al., 2015). - https://towardsdatascience.com/sentence-embedding-3053db22ea77
# 
# ## 2. Dimension Reduction
# 
# * Employed the approach of UMAP for dimension reduction from 4096 to 2D, arguably better than PCA
# 
# ## 3. Clustering
# 
# * Employed the approach of hdbScan as a clustering algo
# 
# **Credits**:
# The data preprocessing codes were taken from https://www.kaggle.com/maksimeren/covid-19-literature-clustering, to convert the JSON files into dataframes*

# In[ ]:


ls


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'
root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()


# In[ ]:


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


# In[ ]:


def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            return data
#             data = data + "<br>" + words[i]
#             total_chars = 0
        else:
            data = data + " " + words[i]
    return data


# In[ ]:


from ast import literal_eval

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 50:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:50]
        summary = get_breaks(' '.join(info), 50)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 50)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
#     try:
#         # if more than one author
#         authors = literal_eval(meta_data['authors'].values[0])
#         if len(authors) > 2:
#             # more than 2 authors, may be problem when plotting, so take first 2 append with ...
#             dict_['authors'].append(". ".join(authors[:2]) + "...")
#         else:
#             # authors will fit in plot
#             dict_['authors'].append(". ".join(authors))
#     except Exception as e:
#         # if only one author - or Null valie
#         dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
#     try:
#         title = get_breaks(meta_data['title'].values[0], 40)
#         dict_['title'].append(title)
#     # if title was not provided
#     except Exception as e:
#         dict_['title'].append(meta_data['title'].values[0])
    
#     # add the journal information
#     dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'abstract_summary'])
df_covid.head()


# In[ ]:


df_covid['body_text'][6]


# # Get all the paragraphs from abstract as a list to process (Body text too much to process..)

# In[ ]:


list_covid = df_covid['abstract'].tolist()


# In[ ]:


# list_covid[0]


# In[ ]:


cd ..


# # Download relevant modules for InferSent

# In[ ]:


mkdir fastText


# In[ ]:


get_ipython().system('curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')


# In[ ]:


mkdir encoder


# In[ ]:


get_ipython().system('unzip fastText/crawl-300d-2M.vec.zip -d fastText/')


# In[ ]:


get_ipython().system('curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl')


# In[ ]:


get_ipython().system('git clone https://github.com/facebookresearch/InferSent')


# # Extract relevant tools needed for Infersent

# In[ ]:


import numpy as np
import torch
from InferSent.models import InferSent
import torch
model_version = 2
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model
# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)
# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)
# Load some sentences


# To compose sentence representations into a paragraph representation, we simply sum the InferSent representations of all the sentences in the paragraph. This approach is inspired by the sum of word representations as composition function for forming sentence representations (Iyyer et al., 2015).

# In[ ]:


subset_list_covid = list_covid
targets = []
new_subset_list_covid = []
embeddings_summed = []

from nltk.tokenize import sent_tokenize
for idx,each in enumerate(subset_list_covid):
    if (len(each) != 0):
        each = each[:int(len(each)/2)] #reduce workload..
    for sent in sent_tokenize(each):
        new_subset_list_covid.append(sent) 
    if len(new_subset_list_covid) == 0 : new_subset_list_covid.append('empty') 
    embeddings = model.encode(new_subset_list_covid, bsize=128, tokenize=False, verbose=True)
    embeddings = [sum(x)/len(x) for x in zip(*embeddings)]
    embeddings_summed.append(embeddings)
    new_subset_list_covid = []
    targets.append(idx)


# In[ ]:


from nltk.tokenize import sent_tokenize
ran = ' '. join(x) for 
sent_tokenize()


# In[ ]:


len(embeddings_summed)


# # Function for cosine distance measure

# In[ ]:


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# # UMAP - Reduce Dimension from 4096 to 2 for visualization

# In[ ]:


get_ipython().system('pip install umap')


# In[ ]:


import umap.umap_ as umap
reducer = umap.UMAP(n_neighbors = 5)
umap_embedding = reducer.fit_transform(embeddings_summed)
umap_embedding.shape


# In[ ]:


print (umap_embedding)


# In[ ]:


# !pip install seaborn


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection COVID-19 Dataset -using infersent - first 100 documents', fontsize=24);


# # Clustering using hdbscan

# In[ ]:


get_ipython().system('pip install hdbscan')


# For clustering algorithm, i chose hdbscan.. it's one of the better ones..

# In[ ]:


import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=2).fit(umap_embedding)
color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*umap_embedding.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.8)


# 

# In[ ]:


subset_list_covid = list_covid[:4]
# subset_list_covid.append(irrelevant_article)
new_subset_list_covid = []
targets = []
embeddings_summed2 = []

from nltk.tokenize import sent_tokenize
for idx,each in enumerate(subset_list_covid):
    for sent in sent_tokenize(each):
        new_subset_list_covid.append(sent)
    embeddings = model.encode(new_subset_list_covid, bsize=128, tokenize=False, verbose=True)
    embeddings = [sum(x)/len(x) for x in zip(*embeddings)]
    embeddings_summed2.append(embeddings)
    new_subset_list_covid = []
    targets.append(idx)


# In[ ]:


list_covid[1]


# In[ ]:


list_covid[4]


# In[ ]:


cosine(umap_embedding[1],umap_embedding[4])

