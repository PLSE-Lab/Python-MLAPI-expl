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

import os

json_files = []
IGN = ['metadata.csv', 'json_schema.txt', 'metadata.readme', 'COVID.DATA.LIC.AGMT.pdf']
for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge'):
    for filename in filenames:
        if not filename in IGN:
            json_files.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pip install -U sentence-transformers


# In[ ]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')


# In[ ]:


import glob
import json
import pandas as pd
from tqdm import tqdm
root_path = '/kaggle/input/CORD-19-research-challenge'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)


# In[ ]:


metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})


# In[ ]:


class COVIDReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)


# In[ ]:



def breaking(txt, length):
    data = ""
    words = txt.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


# In[ ]:


shm = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [],
      'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print("processing index: {}".format(idx))
    content = COVIDReader(entry)
    
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    if len(meta_data) == 0:
        continue
        
    shm['paper_id'].append(content.paper_id)
    shm['abstract'].append(content.abstract)
    shm['body_text'].append(content.body_text)
    
    if len(content.abstract) == 0:
        shm['abstract_summary'].append("Not provided.")
        
    elif len(content.abstract.split(" ")) > 100:
        info = content.abstract.split(' ')[:100]
        summary = breaking(' '.join(info), 40)
        shm['abstract_summary'].append(summary + "...")
    else:
        summary = breaking(content.abstract, 40)
        shm['abstract_summary'].append(summary)
        
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            shm['authors'].append(". ".join(authors[:2]) + "...")
        else:
            shm['authors'].append(". ".join(authors))
    except Exception as e:
        shm['authors'].append(meta_data['authors'].values[0])
    
    try:
        title = breaking(meta_data['title'].values[0], 40)
        shm['title'].append(title)
    except Exception as e:
        shm['title'].append(meta_data['title'].values[0])
    
    shm['journal'].append(meta_data['journal'].values[0])
    
    
df_covid = pd.DataFrame(shm, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()


# In[ ]:


df_covid['body_text'] = df_covid['body_text'].apply(lambda x: x.lower())


# In[ ]:


df_covid['abstract'] = df_covid['abstract'].apply(lambda x: x.lower())


# In[ ]:


df_covid.to_csv("/kaggle/df_covid.csv")


# In[ ]:


df_covid


# In[ ]:


df_covid.columns


# In[ ]:


text = df_covid.drop(["authors", "journal"], axis=1)
text_dict = text.to_dict()

len_text = len(text_dict["paper_id"])


# In[ ]:


paper_id_list  = []
body_text_list = []

title_list = []
abstract_list = []
abstract_summary_list = []
for i in tqdm(range(0,len_text)):
    paper_id = text_dict["paper_id"][i]
    body_text = text_dict["body_text"][i].split("\n")
    title = text_dict["title"][i]
    abstract = text_dict["abstract"][i]
    abstract_summary = text_dict["abstract_summary"][i]
    for b in body_text:
        paper_id_list.append(paper_id)
        body_text_list.append(b)
        title_list.append(title)
        abstract_list.append(abstract)
        abstract_summary_list.append(abstract_summary)


# In[ ]:


df_sentences = pd.DataFrame({"paper_id":paper_id_list},index=body_text_list)


# In[ ]:


df_sentences = pd.DataFrame({"paper_id":paper_id_list,"title":title_list,"abstract":abstract_list,"abstract_summary":abstract_summary_list},index=body_text_list)
df_sentences


# In[ ]:


df_sentences = df_sentences["paper_id"].to_dict()
df_sentences_list = list(df_sentences.keys())
len(df_sentences_list)


# In[ ]:


df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


# In[ ]:


from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
embedder = SentenceTransformer('bert-base-nli-mean-tokens')


# In[ ]:


get_ipython().system('pip install flair')


# In[ ]:


from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import torch
import pickle as pkl



bm = BertEmbeddings()


# In[ ]:


device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# In[ ]:


sentence = Sentence('Test to get embedding size')
bm.embed(sentence)
for token in sentence:
    print(token.embedding)
                
                    
                
z = token.embedding.size()[0]

s = torch.zeros(0,z).to(device)

for paragraph in tqdm(df_sentences_list):
    w = torch.zeros(0, z).to(device)
    para = Sentence(paragraph)
    bm.embed(sentence)
    for token in para:
        w = torch.cat((w, token.embedding.view(-1,z)), 0).to(device)
    s = torch.cat((s, w.mean(dim=0).view(-1, z)))
    
torch.save(s, "embedded-docs.pt")

with open("tensor-pickle.pkl", "wb") as fout:
    pkl.dump(s, fout, protocol=pkl.HIGHEST_PROTOCOL)
    


# In[ ]:




