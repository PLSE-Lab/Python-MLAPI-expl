#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install rake-nltk')


# In[ ]:


from tqdm import tqdm
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import gensim
import spacy
import os
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.matcher import Matcher 
from collections import  Counter
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from rake_nltk import Rake
import tensorflow as tf


# In[ ]:


path="../input/CORD-19-research-challenge/"
data=pd.read_csv(path+"metadata.csv")


# Checking for missing data

# In[ ]:


data.isna().sum()


# In[ ]:


stop=set(stopwords.words('english'))

def build_corpus(df,col="title"):
    corpus=[]
    lem=WordNetLemmatizer()
    stop=set(stopwords.words('english'))
    new= df[col].dropna().str.split()
    new=new.values.tolist()
    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]
    
    return corpus


# Most common words in the abstracts

# In[ ]:


x1=[]
x2=[]
corpus= build_corpus(data, "abstract")
counter= Counter(corpus)
common= counter.most_common()
for word, count in common[:10]:
    if(word not in stop):
        x1.append(word)
        x2.append(count)

sns.barplot(x=x1, y=x2)


# In[ ]:


def prepare_similarity(vectors):
    similarity=cosine_similarity(vectors)
    return similarity

def get_top_similar(sentence, sentence_list, similarity_matrix, topN):
    # find the index of sentence in list
    index = sentence_list.index(sentence)
    # get the corresponding row in similarity matrix
    similarity_row = np.array(similarity_matrix[index, :])
    # get the indices of top similar
    indices = similarity_row.argsort()[-topN:][::-1]
    return [(i,sentence_list[i]) for i in indices]


# In[ ]:


path="../input/cord-19-eda-parse-json-and-generate-clean-csv/"


# In[ ]:


clean_comm=pd.read_csv(path+"clean_comm_use.csv",nrows=5000)
clean_comm['source']='clean_comm'
biox = pd.read_csv(path+"biorxiv_clean.csv")
biox['source']='biorx'

articles=pd.concat([biox,clean_comm])


# In[ ]:


del biox,clean_comm
gc.collect()


# In[ ]:


articles.shape


# In[ ]:


tasks=["What is known about transmission, incubation, and environmental stability",
      "What do we know about COVID-19 risk factors",
      "What do we know about virus genetics, origin, and evolution",
      "What do we know about vaccines and therapeutics",
      "What do we know about non-pharmaceutical interventions",
      "What do we know about diagnostics and surveillance",
      "What has been published about ethical and social science considerations",
      "Role of the environment in transmission",
      "Range of incubation periods for the disease in humans",
      "Prevalence of asymptomatic shedding and transmission",
      "Seasonality of transmission",
      "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)",
      "Susceptibility of populations",
      "Public health mitigation measures that could be effective for control",
      "Transmission dynamics of the virus",
      "Evidence that livestock could be infected",
      "Socioeconomic and behavioral risk factors for this spill-over",
      "Sustainable risk reduction strategies",
      "Resources to support skilled nursing facilities and long term care facilities",
      "Mobilization of surge medical staff to address shortages in overwhelmed communities"]


# In[ ]:


task_df=pd.DataFrame({'title':tasks,'source':'task'})


# In[ ]:


task_df.head()


# In[ ]:


module_url = "../input/universalsentenceencoderlarge4" 
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.load(module_url)


# Appending articles from all the sources

# In[ ]:


articles=pd.concat([articles,task_df])
articles.fillna("Unknown",inplace=True)


# Finding related research papers

# In[ ]:


sentence_list=articles.title.values.tolist()
embed_vectors=embed(sentence_list)['outputs'].numpy()
similarity_matrix=prepare_similarity(embed_vectors)


# In[ ]:


sentence= "Role of the environment in transmission"
similar=get_top_similar(sentence,sentence_list,similarity_matrix,10)


# In[ ]:


for sent in similar:
    print(sent[1])


# Cleaning and storing abstracts from related articles

# In[ ]:


ind,title=list(map(list,zip(*similar)))
titles=[]
texts=[]
for i in ind:
    titles.append(articles.iloc[i]['title'])
    texts.append(articles.iloc[i]['abstract'])


# In[ ]:


import re
def clean(txt):
    txt=re.sub(r'\n','',txt)
    txt=re.sub(r'\([^()]*\)','',txt)
    txt=re.sub(r'https?:\S+\sdoi','',txt)
    return txt


# In[ ]:


texts=list(map(clean,texts))
text_list=' '.join(texts)


# **Constructing knowldege graphs**

# In[ ]:


import spacy
nlp=spacy.load('en_core_web_sm')


# In[ ]:


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

  #############################################################
  
    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
          # check: token is a compound word or not
          if tok.dep_ == "compound":
            prefix = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                   prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
              modifier = prv_tok_text + " "+ tok.text

          ## chunk 3
        if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""      

          ## chunk 4
        if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text

          ## chunk 5  
          # update variables
        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text
  #############################################################

    return [ent1.strip(), ent2.strip()]


# In[ ]:


def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", None, pattern) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)


# ### <font size='3' color='blue'>What is the role of environment in transmission?
# </font>

# To answer the question "**What is the role of environment in transmission?**", we prepare a dataframe that contains subject, relation and object from these abstracts to plot the knowledge graph. 

# In[ ]:


def prepare_df(text_list):
    doc=nlp(text_list)
    df=pd.DataFrame()
    for sent in list(doc.sents):
        sub,obj = get_entities(str(sent))
        relation= get_relation(str(sent))

        if ((len(relation)>2) & (len(sub)>2) &(len(obj)>2)):
            df=df.append({'subject':sub,'relation':relation,'object':obj},ignore_index=True)

    return df


# In[ ]:


df = prepare_df(text_list[24:])
df.head()


# In[ ]:



def draw_kg(pairs,c1='red',c2='blue',c3='yellow'):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
            create_using=nx.MultiDiGraph())
  
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(50, 40), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color=c1,
        edgecolors=c2,
        node_color=c3,
        )
    labels = dict(zip(list(zip(pairs.subject, pairs.object)),
                  pairs['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()


# In[ ]:


draw_kg(df)


# ### <font size='3' color='blue'>What is known about transmission, incubation, and environmental stability?
# </font>

# First we get similar articles

# In[ ]:


sentence= "What is known about transmission, incubation, and environmental stability"
similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)


# Then we prepare the title and abstract

# In[ ]:


ind,title=list(map(list,zip(*similar)))
titles=[]
texts=[]
for i in ind:
    titles.append(articles.iloc[i]['title'])
    texts.append(articles.iloc[i]['abstract'])


# In[ ]:


texts=list(map(clean,texts))
text_list=' '.join(texts)


# Finally we find the subject, object and relations and plot the knowledge graph

# In[ ]:


df = prepare_df(text_list)
draw_kg(df)


# ### <font size='3' color='blue'>What do we know about virus genetics, origin, and evolution?
# </font>

# In[ ]:


sentence= "What do we know about virus genetics, origin, and evolution"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)

ind,title=list(map(list,zip(*similar)))
titles=[]
texts=[]
for i in ind:
    titles.append(articles.iloc[i]['title'])
    texts.append(articles.iloc[i]['abstract'])
    
texts=list(map(clean,texts))
text_list=' '.join(texts)


# In[ ]:


df = prepare_df(text_list)
draw_kg(df)


# ### <font size='3' color='blue'>What is known about the range of incubation periods for the disease in humans?
# </font>

# In[ ]:


sentence="Range of incubation periods for the disease in humans"
similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)

ind,title=list(map(list,zip(*similar)))
titles=[]
texts=[]
for i in ind:
    titles.append(articles.iloc[i]['title'])
    texts.append(articles.iloc[i]['abstract'])
    
texts=list(map(clean,texts))
text_list=' '.join(texts)


# In[ ]:


df = prepare_df(text_list)
draw_kg(df)


# In[ ]:




