#!/usr/bin/env python
# coding: utf-8

# ### Created by a [TransUnion](www.transunion.com) data scientist that believes that information can be used to change our world for the better. #InformationForGood**
# 
# ### Our goal here is to summarize abstracts that are related to  covid19 risk factors using text rank. Joint work with [Karen](https://www.kaggle.com/kejinqian/find-answers-using-lda-and-skip-thoughts ).
# 
# # COVID-19: extractive text summarization using text rank 

# ### Pipeline
# 
# 1. For each subtopic(smoking and pulmonary diseases, pregnancy and neonates, co-infections and commorbidities, etc)  generate a new dataframe for related abstracts using syntax. [These syntax or keyword were found using an LDA model output from karen's link here](https://www.kaggle.com/kejinqian/find-answers-using-lda-and-skip-thoughts )
# 2. For each abstract   taken [from here](kkk.com), perform an extractive text summarization (text rank) using the top 3 sentences
#     1. Tokenize into sentences and cleaning,
#     2. Create sentence representations  using GLOVE word embeddings,
#     3. Construct similarity matrix using cosine similarity,
#     4. Build sentences network and apply page rank to find an importance score fro each sentence
#     5. Select top 3 sentences to build the new summary.

# In[ ]:


import numpy as np
import pandas as pd
import os
import nltk
nltk.download('punkt')
import re
import networkx as nx
from termcolor import colored 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['abstract','background','summary','introduction'])


# ## On Smoking

# In[ ]:


smoke=pd.read_csv("../input/subtopic/smokpaper.csv",sep=',')
df=smoke.loc[[0]]
df


# In[ ]:


## tokenize sentences
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['abstract']:
    sentences.append(sent_tokenize(s))

sentences= [y for x in sentences for y in x] 
sentences[:2]


# In[ ]:


## word embed from glove 
word_embedings = {}
f= open('../input/glovewordembed/glove.6B.100d.txt',encoding='utf-8')

for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    word_embedings[word]= coefs
f.close()  


# In[ ]:


## remove special char
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


# In[ ]:


clean_sentences[:2]


# In[ ]:


## remove stop words
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# In[ ]:


clean_sentences[:2]


# In[ ]:


##sentence to vectors using word embeddings
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)


# In[ ]:


sentence_vectors[:1]


# In[ ]:


# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# In[ ]:


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
## each node represents a sentence
nx.draw(nx_graph,pos=nx.spring_layout(nx_graph),with_labels = True)
nx_graph


# ### Score and sentence

# In[ ]:


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')
summary=summary.rename(columns={0:'score',1:"sentence"})
summary


# ### Top 3 sentences to create new summary
# 

# In[ ]:


ranked= summary['sentence'].values.tolist()
#top 20
#for i in range(len(ranked)):
  #print(i+1,")", ranked[i], "\n")
 
#print(colored(list(smoke['abstract']) ,'green'))
#first=sentences[0]
#aa=ranked[:3]
#aa.append(first)
#'.'.join(aa)   ## the first sentence  may be usefull

print(colored(list(df['abstract']) ,'green'))
'.'.join(ranked[:3])


# ## On pregancy

# In[ ]:


preg= pd.read_csv("../input/subtopic/pregnantpaper.csv",sep=',')
df2=preg.loc[[10]]
df2
from nltk.tokenize import sent_tokenize
sentences = []
for s in df2['abstract']:
    sentences.append(sent_tokenize(s))

sentences= [y for x in sentences for y in x]  

## remove special char
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)    

## ranking 
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')
summary=summary.rename(columns={0:'score',1:"text"})

## output
summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')
summary=summary.rename(columns={0:'score',1:"text"})

ranked= summary['text'].values.tolist()
#top 20
#for i in range(len(ranked)):
  #print(i+1,")", ranked[i], "\n")
#for i in range(4):    
    #print( ranked[i])
print(colored(list(df2['abstract']) ,'green'))
'.'.join(ranked[:3])


# ### On cardiovascular and cerebrovascular

# In[ ]:


card= pd.read_csv("../input/subtopic/respiratory_cardio_paper.csv",sep=',')
df3=card.loc[[2]]
from nltk.tokenize import sent_tokenize
sentences = []
for s in df3['abstract']:
    sentences.append(sent_tokenize(s))

sentences= [y for x in sentences for y in x]  

## remove special char
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)    

## ranking 
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')
summary=summary.rename(columns={0:'score',1:"text"})

## output
summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')
summary=summary.rename(columns={0:'score',1:"text"})

ranked= summary['text'].values.tolist()
#top 20
#for i in range(len(ranked)):
  #print(i+1,")", ranked[i], "\n")
#for i in range(4):    
    #print( ranked[i])
print(colored(list(df3['abstract']) ,'green'))
'.'.join(ranked[:3])


# **Finally, we could just encapsulate all this into a function that receives as inputs an abstract and outputs a summary**
