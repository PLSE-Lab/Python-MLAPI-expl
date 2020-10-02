#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Objective : To build a small knowledge graph that will contain structured information extracted from unstructured text.**

# In[ ]:


import spacy
import re
import string
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import nltk
def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]

def printToken(token):
    print(token.text, "->", token.dep_)
    
def appendChunk(original, chunk):
    return original + ' ' + chunk
def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)
def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)
def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''
            
    print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())


def processSentence(sentence):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)


# ## Information extraction and knowledge graphs
# 
# * Information extraction is a technique of extracting structured information from unstructured text. This means taking a raw text(say an article) and processing it in such way that we can extract information from it in a format that a computer understands and can use. This is a very difficult problem in NLP because human language is so complex and lots of words can have a different meaning when we put it in a different context.
# 
# * Information extraction consists of several, more focused subfields, each of them having difficult problems to solve. For example, here's an approach to using information extraction techniques for performing named entity recognition. Another subfield which has gained much interest from the community is keywords extraction.
# 
# ## Graph
# 
# * A knowledge graph is a way of storing data that resulted from an information extraction task. Many basic implementations of knowledge graphs make use of a concept we call triple, that is a set of three items(a subject, a predicate and an object) that we can use to store information about something.

# In[ ]:



df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8},nrows=100)
df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16},nrows=100)


# In[ ]:


df_train.head()


# In[ ]:


def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    pos = nx.spring_layout(G,k=2, iterations=50)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=5000, node_color='lightblue', alpha=0.9,font_size=10,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()


# In[ ]:


def clean_text(x):
    text = re.sub('(\d+)','',x)   
    text = text.lower()
    return text
def remove_url(x):
    text = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})\/([a-zA-Z0-9_]+]*)',' ',x)
    return text
def remove_punct(x):
    text_without_puct = [t for t in x if t not in string.punctuation]
    text_without_puct = ''.join(text_without_puct)
    return text_without_puct
stop_words = nltk.corpus.stopwords.words('english')
from nltk.corpus import stopwords
stop = stopwords.words('english')

df_train['text'] = df_train['text'].apply(clean_text)
df_train['text'] = df_train['text'].apply(remove_url)
df_train['text'] = df_train['text'].apply(remove_punct)
df_train['text'] = df_train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# * Import our dependencies
# * Use spaCy to split the text in sentences
# * For each sentence, use spaCy to figure out what kind of word is every word in that sentence: is it a subject, an object, a predicate and so on
# * Use the information from above to figure out where in the triple we should put a word
# * Finally build the triples
# * Build and show the knowledge graph

# In[ ]:


#nlp = spacy.load('en')
#df_train['text'] = df_train['text'].apply(nlp)
text = df_train['text'].tolist()
read_1 = text[:10]
read_1


# In[ ]:


if __name__ == "__main__":
    
    text = 'deeds reason earthquake may allah forgive us'

    sentences = getSentences(text)
    nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print (text)
    for sentence in sentences:
        triples.append(processSentence(sentence))


# In[ ]:


printGraph(triples)


# In[ ]:


if __name__ == "__main__":
    
    text = 'people receive wildfires evacuation orders california'

    sentences = getSentences(text)
    nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print (text)
    for sentence in sentences:
        triples.append(processSentence(sentence))


# In[ ]:


printGraph(triples)


# In[ ]:


if __name__ == "__main__":
    
    text = 'flood disaster heavy rain causes flash flooding streets manitou colorado springs areas'

    sentences = getSentences(text)
    nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print (text)
    for sentence in sentences:
        triples.append(processSentence(sentence))


# In[ ]:


printGraph(triples)


# Based on - https://programmerbackpack.com/python-nlp-tutorial-information-extraction-and-knowledge-graphs/
