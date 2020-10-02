#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flair')
get_ipython().system('pip install transformers')
get_ipython().system('pip install node2vec')
get_ipython().system('pip install wordcloud #https://github.com/amueller/word_cloud')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from flair.models import SequenceTagger
from flair.data import Sentence
from transformers import pipeline
pd.set_option('display.max_columns', 30)
from collections import Counter 
import nltk
import matplotlib.pyplot as plt
import re
from tqdm.autonotebook import tqdm
import networkx as nx
from wordcloud import WordCloud
from sklearn.cluster import DBSCAN

import plotly.graph_objects as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def tokenizer(text):
    text = text.split(',')
    res = []
    for t in text:
        res.extend(nltk.word_tokenize(t))
        
    return res


# In[ ]:


data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
data.head()


# In[ ]:


text = data.summary.dropna().values
print(f"Text lenght: {len(text)}")


# In[ ]:


symptoms = data.symptom.dropna().tolist()
print(len(symptoms))


# In[ ]:


print(f"Total dataset len: {len(data)}")
print(f"Death: {len(data.loc[data.death == '1'])}")
print(f"Recovered: {len(data.loc[data.recovered == '1'])}")


# # NLP

# ## Question answering

# We can collect some data using NLP question answering. Adding more questions, and filtering by confidesce score we can fill some missing walues

# In[ ]:


ner_tagger = SequenceTagger.load('ner')

nlp_qa = pipeline('question-answering')


# Look like NER is not useful for us, if we want parse dates its better to use regular expressions.

# In[ ]:


match = re.findall(r'(\d+/\d+/\d+)',text[1])
print(match)


# In[ ]:


sentence = Sentence(text[1])
ner_tagger.predict(sentence)
print(sentence.to_tagged_string())


# Try to use question answering. 
# Some sample questions

# In[ ]:


questions = [
    'How old?',
    'Gender?',
    'Where from?',
    'When symptoms onset?',
    'When Hospitalized?',
    'When quarantined?',
]


# In[ ]:


print(text[0])
for q in questions:
    print(f"Question: {q}, answer: {nlp_qa(context = text[0], question=q)}")


# Wordcloud basic

# In[ ]:


wordcloud = WordCloud().generate(' '.join(symptoms))

plt.figure(figsize=[10, 6])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[ ]:


from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentRNNEmbeddings([glove_embedding])


# In[ ]:


embeddings = np.empty((len(text), 128))
for idx, t in tqdm(enumerate(text)):
    sentence = Sentence(t)
    document_embeddings.embed(sentence)
    embeddings[idx, :] = (sentence.get_embedding().detach().numpy())


# ## Clustering

# In[ ]:


dbscan=DBSCAN(eps=0.08, min_samples=2,metric='cosine' ).fit(embeddings)

df_cluster = pd.DataFrame({"text":text, "cluster":dbscan.labels_})


# In[ ]:


df_cluster.cluster.value_counts()


# In[ ]:


df_cluster.loc[df_cluster['cluster'] == 31].head()


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_pca(train,y, text="Plot", algo = 'TSNE', size = 2):
    """Function visualizating PCA/TSNE"""

    plt.figure(figsize=(20,8))
    if algo == 'PCA':
        pca = PCA(n_components = 2,copy=False)
    elif algo == 'TSNE':
        pca = TSNE(n_components = 2)
    else:
        print('Unknown algo, using PCA...')
        pca = PCA(n_components = 2, copy=False)
        
    train_pca = pca.fit_transform(train)

    plt.scatter(train_pca[:,0], train_pca[:,1],c=y, edgecolor='none', alpha=0.9,
            cmap=plt.cm.get_cmap('seismic', size))
    plt.title(text)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()


# In[ ]:


np.unique(dbscan.labels_)


# In[ ]:


plot_pca(embeddings, dbscan.labels_, size = len(np.unique(dbscan.labels_)))


# In[ ]:


'''plt.figure(figsize=[10, 8])
plt.title("Text embedding")
plt.scatter(transformed[:,0], transformed[:,1], edgecolor='none', alpha=0.9,)
plt.xlabel('X-comp')
plt.ylabel('Y-comp')
plt.show()'''


# Look like we have some clusters at document embedding space

# # Symptoms Node2Vec

# Create symptoms graph and Node2Vec model for finding similar symptom and get a probability of next potential symptoms

# In[ ]:


symptoms[:10]


# In[ ]:


counter = Counter()
links = dict(dict())

for row in symptoms:
    
    row = tokenizer(row)    
    
    for symptome in row:
        counter[symptome] += 1
        
    for subsymptome in row:
        if not(symptome in links.keys()):
            links[symptome] = dict()
            
        if not(subsymptome in links[symptome].keys()):
            links[symptome][subsymptome] = 0
            
        if symptome != subsymptome:    
            links[symptome][subsymptome] += 1
            
for key1 in links.keys():
    for key2 in links[key1].keys():
        links[key1][key2] /= counter[key1]
        
links_dict = dict()
for key1, val1 in links.items():
    for key2, val2 in links[key1].items():
        #if val2 >= 0.5:
        if key1 != key2:
            links_dict[(key1, key2)] = val2
        


# In[ ]:


counter.most_common(10)


# In[ ]:


size_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
size_df.columns = ['label', 'size']

links_df = pd.DataFrame([[list(key)[0], val, list(key)[1]] for key, val in links_dict.items() if list(key)[0] != list(key)[1]], columns = ['source', 'weight', 'target'])


# In[ ]:


G = nx.Graph()
for idx, row in size_df.iterrows():
    G.add_node(row['label'], size = float(row['size']))

for idx, row in links_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=float(row['weight']))


# In[ ]:


print(f"Nodes len: {len(list(G.nodes()))}")
print(f"Edges len: {len(list(G.edges()))}")
# %%
plt.figure(figsize=[20, 8])

params = {
    'edge_color'    : '#FFDEA2',
    'width'         : 1,
    'with_label'    : True,
    'font_weight'   : 'regular'
}

node_list = [i*10 for i in size_df['label'].values]
node_size = [i*10 for i in size_df['size'].values]
nx.draw_networkx(G,node_list = node_list,node_size=node_size, **params)


# Create node2vec for symptomes prediction.

# In[ ]:


from node2vec import Node2Vec

n2v = Node2Vec(G, dimensions=15, num_walks=100, workers=4)
node_model = n2v.fit(size=3, window=2, seed=42, iter=1, sg=1)


# In[ ]:


node_model.most_similar(['fever'], topn=15)


# In[ ]:


node_model.most_similar(['fever', 'cough'], topn=15)


# In[ ]:




