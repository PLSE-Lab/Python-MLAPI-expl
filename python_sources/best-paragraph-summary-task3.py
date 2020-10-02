#!/usr/bin/env python
# coding: utf-8

# ## Installing and testing sentence embedding extraction modules. Testing For GPU and setting up device. 
# ### Using Pytorch based Bio-Bert download via biobert-embedding
# 
# ### **Testing below package to generate embedding** 
# [Biobert Reference](https://github.com/Overfitter/biobert_embedding)
# **This package main code is modified to run it on GPU**
# 
# [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
# 
# #### Reserch paper for the search is selected based on Clustering Via Citation and other networks 
# 
# 

# In[ ]:



import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import subprocess
import pickle
import numpy as np
import io
from sklearn.metrics.pairwise import cosine_similarity
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pprint
import matplotlib.pyplot as plt
import pickle as pkl
get_ipython().system('pip install biobert-embedding')
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)
from nltk import tokenize


# # Import package and change code to run on GPU
# 
# 
# 

# In[ ]:


from biobert_embedding.embedding import BiobertEmbedding
import os
import torch
import logging
import tensorflow as tf
from pathlib import Path
from biobert_embedding import downloader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
class BiobertEmbedding(object):
    """
    Encoding from BioBERT model (BERT finetuned on PubMed articles).
    Parameters
    ----------
    model : str, default Biobert.
            pre-trained BERT model
    """

    def __init__(self, model_path=None):

        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = downloader.get_BioBert("google drive")

        self.tokens = ""
        self.sentence_tokens = ""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.model_path)
        self.model.to(device)
        logger.info("Initialization Done !!")

    def process_text(self, text):

        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        return tokenized_text


    def handle_oov(self, tokenized_text, word_embeddings):
        embeddings = []
        tokens = []
        oov_len = 1
        for token,word_embedding in zip(tokenized_text, word_embeddings):
            if token.startswith('##'):
                token = token[2:]
                tokens[-1] += token
                oov_len += 1
                embeddings[-1] += word_embedding
            else:
                if oov_len > 1:
                    embeddings[-1] /= oov_len
                tokens.append(token)
                embeddings.append(word_embedding)
        return tokens,embeddings


    def eval_fwdprop_biobert(self, tokenized_text):

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        return encoded_layers


    def word_vector(self, text, handle_oov=True, filter_extra_tokens=True):

        tokenized_text = self.process_text(text)

        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # Stores the token vectors, with shape [22 x 768]
        word_embeddings = []
        logger.info("Summing last 4 layers for each token")
        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            word_embeddings.append(sum_vec)

        self.tokens = tokenized_text
        if filter_extra_tokens:
            # filter_spec_tokens: filter [CLS], [SEP] tokens.
            word_embeddings = word_embeddings[1:-1]
            self.tokens = tokenized_text[1:-1]

        if handle_oov:
            self.tokens, word_embeddings = self.handle_oov(self.tokens,word_embeddings)
        logger.info(self.tokens)
        logger.info("Shape of Word Embeddings = %s",str(len(word_embeddings)))
        return word_embeddings



    def sentence_vector(self,text):

        logger.info("Taking last layer embedding of each word.")
        logger.info("Mean of all words for sentence embedding.")
        tokenized_text = self.process_text(text)
        self.sentence_tokens = tokenized_text
        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

        # `encoded_layers` has shape [12 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = encoded_layers[11][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        logger.info("Shape of Sentence Embeddings = %s",str(len(sentence_embedding)))
        return sentence_embedding


# # Check GPU

# In[ ]:


device.type


# # Sentence transformer

# In[ ]:


pip install -U sentence-transformers


# # Download BioBert

# In[ ]:


model_path = downloader.get_BioBert("google drive")


# # setting SentenceTransformer

# In[ ]:


# Use BERT for mapping tokens to embeddings
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
word_embedding_model = models.BERT('/kaggle/working/'+model_path.name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# # Read papers Selected for Task3 

# In[ ]:


data=pd.read_excel("/kaggle/input/task3covid/task3_results_summary.xlsx",index=False).dropna()
selected_papers_task3=pd.read_csv('/kaggle/input/task3-results/task3_results.csv')

# Select query or subtask field
query_subtask="Queries"
# query_subtask="Subtask mapping"

# Select summary or original text field
# summary_field="summary"
summary_field="Text"


# # Data preparation

# In[ ]:


selected_papers_task3=selected_papers_task3.drop_duplicates(subset=[query_subtask,'cord_uid'])
selected_papers_task3[query_subtask]=selected_papers_task3[query_subtask].apply(lambda x:preprocess_sentence(x))

data=data[~data['Name'].isin(['TITLE','ABSTRACT'])]
data=data.drop_duplicates(subset =summary_field)
meta_df_title_abstract=data
len1=meta_df_title_abstract.shape[0]
list1=list(range(len1))
meta_df_title_abstract['pid']=list1
meta_df_title_abstract.head()
# meta_df_title_abstract['summary_preprocessed']=meta_df_title_abstract['Text'].apply(lambda x:tokenize.sent_tokenize(x))
# new_data_sent=meta_df_title_abstract['summary_preprocessed'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')
# new_data_sent=new_data_sent.merge(meta_df_title_abstract[['cord_uid', 'lsid', 'gsid', 'Name', 'Text', 'Subtype', 'summary', 'pid']],right_index=True,left_index=True,how='left')
meta_df_title_abstract['wrd_cnt']=meta_df_title_abstract[summary_field].str.split().str.len()
new_data_sent_strip=meta_df_title_abstract[meta_df_title_abstract['wrd_cnt']>30]
print("wrd cnt > 30 " + str(new_data_sent_strip.shape))
new_data_sent_strip=new_data_sent_strip[new_data_sent_strip['wrd_cnt']<550]
print("wrd cnt < 600 " + str(new_data_sent_strip.shape))
# new_data_sent_strip['value_edit']=new_data_sent_strip[summary_field].apply(lambda x:preprocess_sentence(x))
new_data_sent_strip[summary_field]=new_data_sent_strip[summary_field].apply(lambda x:preprocess_sentence(x))

query_list=selected_papers_task3[query_subtask].unique().tolist()
new_data_sent_strip=new_data_sent_strip.reset_index()
summaries=new_data_sent_strip[summary_field].tolist()
Comp_reserch_data=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')


# ## Create embedding for queries and summaries 

# In[ ]:


# query_text=['What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?',
# 'Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.',
# 'Prevalence of asymptomatic shedding and transmission',
# 'Seasonality of transmission of covid corona virus.',
# 'Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)',
# 'Disease models, including animal models for infection, disease and transmission',
# 'Immune response and immunity',
# 'Role of the environment in transmission',
# 'Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings',
# 'Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings',
# 'Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).']

import pickle

query_embedding=model.encode(query_list,show_progress_bar=False)
summ_embedding=model.encode(summaries,show_progress_bar=False)

with open('/kaggle/working/query_embedding_sent.pickle', 'wb') as handle:
    pickle.dump(query_embedding, handle)
print(len(query_embedding))
print(query_embedding[0].shape)

with open('/kaggle/working/embeddings37912.pickle', 'wb') as handle:
    pickle.dump(summ_embedding, handle)
print(len(summ_embedding))
print(summ_embedding[0].shape)


# # Run Similarity test

# In[ ]:


import scipy.spatial
new_frame=pd.DataFrame(columns=['query','cord_uid','cord_uid_index','summary','similarity'])

for k in range(len(query_list)):
    query=query_list[k]
    query_emb=list(query_embedding[k])
    cord_id_index=selected_papers_task3.index[selected_papers_task3[query_subtask]==query].tolist()
    cord_id_list=[]
    for j in cord_id_index:
        cid=selected_papers_task3.loc[j,'cord_uid']
        cord_id_list.append(cid)
    cord_id_list_summ=[]
    cord_id_list_emb=[]
    for j in cord_id_list:
        ind_list=new_data_sent_strip.index[new_data_sent_strip['cord_uid']==j].tolist()
        for p in ind_list:
            text=new_data_sent_strip.loc[p,'summary']
            cord_id_list_summ.append(text)
            summm_embed=list(summ_embedding[p])
            cord_id_list_emb.append(summm_embed)
    distances = scipy.spatial.distance.cdist([query_emb], cord_id_list_emb, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    closest_n = min(len(cord_id_list_emb),5)       
    for idx, distance in results[0:closest_n]:
        text1=cord_id_list_summ[idx]
        cord_uid1=new_data_sent_strip.index[new_data_sent_strip['summary']==text1].tolist()[0]
        cid=new_data_sent_strip.loc[cord_uid1,'cord_uid']
        val=1-distance
        new_frame=new_frame.append({'query':query,'cord_uid':cid,'cord_uid_index':cord_uid1,'summary':text1,'similarity':val},ignore_index=True)


# # Display results

# In[ ]:


from IPython.display import display, HTML
#display(HTML(all_search.to_html()))
for query_id in range(len(query_list)):
    display(HTML('<font size="5" color="blue"> <b> Query Searched : </b> </font><p> <font size="4">'+query_list[query_id]+'</font><p>'))
    new_frame1=new_frame[new_frame['query']==query_list[query_id]].drop(columns=['query'])
    new_frame1=new_frame1.merge(Comp_reserch_data[['cord_uid',  'title', 'license', 'publish_time', 'authors', 'journal']],on=['cord_uid'],how='inner')
    new_frame1=new_frame1[['similarity','summary','cord_uid','title', 'license', 'publish_time', 'authors', 'journal']]
    display(HTML(new_frame1.style.set_properties(subset=['summary'],                                              **{'font-weight': 'bold','font-size': '9pt','text-align':"left",'background-color': 'lightgrey','color': 'black'}).set_table_styles(                                             [dict(selector='th', props=[('text-align', 'left'),('font-size', '12pt'),('background-color', 'skyblue'),('border-style','solid'),('border-width','1px')])]).hide_index().render()))

    display(HTML("-------End-----"*15))


# In[ ]:


print("Done")


# In[ ]:




