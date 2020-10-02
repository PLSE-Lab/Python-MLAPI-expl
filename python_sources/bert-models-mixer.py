#!/usr/bin/env python
# coding: utf-8

# # Notebook purpose:
# * At this notebook you can get the nearest articles relevant to your questions about COVID_19.
# * You can get nearest article for each risk factor.
# 

# # Methodology for mixer
# *  we use the output vectors from [our notebook](https://www.kaggle.com/fatma98/biobert-bert-encoding), at which we make BERT and BioBERT vectorization model, and here we will avraging the score of BERT model based on BERT ecoding on the paper body_txt and BioBERT model  based on BioBERT encoding on papers title and abstract.

# # Methodology for LDA's risk factor tester
# * Here we used LDA model to ask queries about our task -Risk Factors-.
# * Each query is fitted on our main kernel and Pkl files are uploaded here for fast use.

# ### Resources:
# * [BioBERT_BERT Encoding notebook](https://www.kaggle.com/jdparsons/biobert-corex-topic-search)
# * [Topic Modeling notebook](https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles)
# * [BERT Word Embeddings Tutorial](https://l.facebook.com/l.php?u=https%3A%2F%2Fmccormickml.com%2F2019%2F05%2F14%2FBERT-word-embeddings-tutorial%2F%3Ffbclid%3DIwAR0Tszxw2niNjbWOYvm9K3NV6syx4kP2AsbFvttIUArZxn0sJ_zGEOIaEF4&h=AT3C0KI7RcUlmwdtb-YKUvyBzhdXo9zIjTM3dwBreUm3XmyVyepLMqwTKnzbj_rmoH_FJa1x64is1L11hGHfDHInnidkbHzimnZyh3Zx4Z4vJQueXowbNHBWLnrkq-zo5hyXGw)

# # Pros. 
# 1. You can try many models:  Biobert, Bert and LDA.
#    Then by avraging the scores together -as in the main kernel-, you can get more accurate output.
# 2. The body is used on some methods and the abstract with the title on others, so as to get as much information as possible when mixing the methods.
# 3. The output of all the methods is saved as pkl files, so the user can use it quickly.
# 4.  Very interactive GUI.
# 5. Visualization of data.
# 

# # Cons
# * Running code is very slow for the first time for new dataset.

# # Imports & Installation

# In[ ]:


get_ipython().run_cell_magic('time', '', "!pip install tensorflow==1.15\n!pip install bert-serving-server==1.10.0\n!pip install bert-serving-client==1.10.0\n\n!cp /kaggle/input/biobert-pretrained /kaggle/working -r\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta\n\n!pip install transformers\n!pip install sentence-transformers\n!pip install rake-nltk\n\nprint('installation done')")


# In[ ]:


import subprocess
import pickle as pkl
import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from ipywidgets import interact, widgets # this is what makes the dataframe interactive
from scipy.spatial.distance import cdist
from scipy.spatial.distance import jensenshannon
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from textblob import TextBlob
import pyLDAvis.gensim
import pyLDAvis
import gensim
import spacy
import os
from scipy import spatial

plt.style.use("dark_background")


# # Data load

# In[ ]:


df = pkl.load(open('../input/bertbiobertdataframe/BERT-BioBERT-dataframe.pkl', "rb"))


# In[ ]:


meta_df=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
meta_df = meta_df.dropna(subset=['url'])


# # Data visualization

# 1.The following plot show you the top ten journals participated to share COVID_19 papers

# In[ ]:


from collections import Counter
meta_df = meta_df.dropna(subset=['journal'])
journals=meta_df['journal'].tolist()

count = Counter(journals)
freq=count.most_common(10)

paper_count=pd.DataFrame(freq,columns=['journals','number of papers'])
paper_count.sort_values('number of papers', ascending=False).set_index('journals')[:20].sort_values('number of papers', ascending=True).plot(kind='barh')


# ### 2.Distribution of length for Both Body and Abstract

# In[ ]:


import seaborn as sns

headline_length=df['title'].str.len()
sns.distplot(headline_length)
plt.show()
headline_length=df['abstract'].str.len()
sns.distplot(headline_length)
plt.show()
headline_length=df['body_text'].str.len()
sns.distplot(headline_length)
plt.show()


# ### 3.Visualisation of The distribution of abstract sentiment polarity score

# In[ ]:


df['polarity'] = df['abstract'].map(lambda text: TextBlob(text).sentiment.polarity)
df['abstract_len'] = df['abstract'].astype(str).apply(len)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df['polarity'].iplot(kind='hist',bins=50,xTitle='polarity',linecolor='black',yTitle='count',title='Sentiment Polarity Distribution')


# # Models

# In[ ]:


#Bert
bert_client = SentenceTransformer('bert-base-nli-max-tokens')

#Biobert
bio_bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=None -max_batch_size=32 -num_worker=2'
process = subprocess.Popen(bio_bert_command.split(), stdout=subprocess.PIPE)
from bert_serving.client import BertClient
biobert_client = BertClient(ignore_all_checks=True)


# In[ ]:


bio_vectors = np.array(df.biobert_vector.tolist())
bert_vectors = np.array(df.bert_vector.tolist())


# ## Cosine function to get models Score:

# In[ ]:


from scipy import spatial
def score(model_vectors,model_encode,size):
    score = []
    for i in range(size):
        result = 1 - spatial.distance.cosine(model_vectors[i],model_encode)
        score.append(result)
    return score


# ## Get avrage Score:

# In[ ]:


def avrage (model1_score,model2_score):
    l=[sum(n) for n in zip(*[model1_score,model2_score])]
    final_score = [x * 0.5 for x in l]
    return final_score 


# # Mixer GUI

# In[ ]:


default_question = 'Neonates and pregnant women'
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('max_colwidth', 180)

results=[]
total_docs=df.shape[0]
@interact
def search_articles(
    query=default_question,
    num_results=[10, 25, 100],show_scores=[False, True],score_type=['Cosine']):

    bio_encode =biobert_client.encode([query])
    bert_encode = bert_client.encode([query])
   
    if score_type is 'Cosine':
        bert_score=score(bert_vectors,bert_encode,total_docs)
        bio_score=score(bio_vectors,bio_encode,total_docs)
        f_score=avrage(bert_score,bio_score)
        df["score"] = f_score
        select_cols = ['title', 'abstract', 'authors', 'score','url']
        results = df[select_cols].sort_values(by=['score'], ascending=False).head(num_results)
        results = results.dropna(subset=['title'])
        
#     print("results : {}".format(results[20703	]))
    if (len(results.index) == 0):
        print('NO RESULTS')
        
        return None
    else:
        

        top_row = results.iloc[0]

        print('TOP RESULT OUT OF ' + str(total_docs) + ' DOCS FOR QUESTION:\n' + query + '\n')
        print('TITLE: ' + str(top_row['title']) + '\n')
        print('ABSTRACT: ' + top_row['abstract'] + '\n')
        #print('PREDICTED TOPIC: ' + topic_list[int(top_row['best_topic'].replace('topic_', ''))])

        print('\nAUTHORS: ' + str(top_row['authors']))

        select_cols.remove('authors')
        
        return results[select_cols]


# ***

# ***
