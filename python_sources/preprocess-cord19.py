#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This is the preprocessing portion of an effor to optimize a Latent Dirichlet Allocation (LDA) topic model.  We break up this effort into multiple steps in order to speed up testing and making changes.  This also uses less memory and is less likely to time out.  
# 
# A topic model learns a topic-feature matrix of abstract topics and features (word or ngrams) and a document-topic matrix of documents and topics, from a document-feature matrix of documents and features.  From this factorization we achieve statistical feature vectors for each topic and topic vectors for each document in the training corpus.  We can then find topic vectors for the questions we would like to ask the corpus of documents.  We will use the closest matching documents in the CORD-19 dataset in an attempt to answer the task questions.  LDA assumes a Dirichlet prior on topic-feature and document-topic distributions.  In other words it assumes each topic is defined by and small collection of words or ngrams and that each documnent consists of a small number of topics 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    print(dirname)
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Additional Imports

# In this section of the code we have additional imports to the standard ones above.  Glob will be used to find all the files we need and json will be used to open the full documents of the CORD-19 dataset.  Gensim and nltk will be used for language preprocessing tasks.  Pickle will be used to write results to output so that we can break up the preprocessing and model learning tasks and speed up testing and applying changes.

# In[ ]:


#additional imports
import glob
import json
import gensim
import nltk
import pickle #for writing to output
import gc


# # Definitions for reading the json files.

# The purpose of these definitions is to read all json files in a path and return a dictionary containing entries for the paper's id, title, abstract, and body text.  These json files are the papers that we have full body text for. 
# 
# Now that we are up to 50k+ documents we keep running out of memory

# In[ ]:


def readarticle(filepath):
    paperdata = {"paper_id" : None, "title" : None, "abstract" : None}
    with open(filepath) as file:
        filedata = json.load(file)
        paperdata["paper_id"] = filedata["paper_id"]
        paperdata["title"] = filedata["metadata"]["title"]
                
        if "abstract" in filedata:
            abstract = []
            for paragraph in filedata["abstract"]:
                abstract.append(paragraph["text"])
            abstract = '\n'.join(abstract)
            paperdata["abstract"] = abstract
            #print(abstract)
        else:
            paperdata["abstract"] = []
            #print("no abstract")

        #body_text = []
        #for paragraph in filedata["body_text"]:
        #    body_text.append(paragraph["text"])
        #body_text = '\n'.join(body_text)
        #paperdata["body_text"] = body_text
        
    return paperdata

def read_multiple(jsonfiles_pathnames):
    papers = {"paper_id" : [], "title" : [], "abstract" : []}
    for filepath in jsonfiles_pathnames:
        paperdata = readarticle(filepath)
        if len(paperdata["abstract"]) > 0: 
            papers["paper_id"].append(paperdata["paper_id"])
            papers["title"].append(paperdata["title"])
            papers["abstract"].append(paperdata["abstract"])
            #papers["body_text"].append(paperdata["body_text"])
            #print("not none")
        #else:
            #print("none")
    print(len(papers["paper_id"]))
    print(len(papers["title"]))
    print(len(papers["abstract"]))
    gc.collect()
    return papers


# # Bigram and Trigram definitions

# Below are definitions for making bigrams (sequences of 2 words) and trigrams (collections of 3 words).  We will be skipping trigrams, but the definition is included in case it is desired later.

# In[ ]:


def make_bigram(tokenized_data, min_count = 5, threshold = 100):
    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
    #after Phrases a Phraser is faster to access
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    gc.collect()
    return bigram

def make_trigram(tokenized_data, min_count = 5, threshold = 100):
    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold = 100)
    #after Phrases a Phraser is faster to access
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    gc.collect()
    return trigram
    


# # Definition for tokenizing documents 

# Reading in the documents was done inside the same function that tokenizes the documents so that the raw documents can go out of scope.  This is done to reduce memory usage.  We will only be looking at papers that have a json file.  We will be saving the resulting dictionary for use later. 

# In[ ]:


def readjson_retbodytext(jsonfiles_pathnames):
    print("reading json files")
    documents = read_multiple(jsonfiles_pathnames)
    print("writing documents dictionary to output for use in another kernel")
    with open("documents_dict.pkl", 'wb') as f:
        pickle.dump(documents, f)
    print("done writing documents dict.  Format is paper_id, title, body_text")
    gc.collect()
    return documents["abstract"]
    
def open_tokenize(jsonfiles_pathnames):
    
    body_text = readjson_retbodytext(jsonfiles_pathnames)
    
    print("removing stopwords, steming, and tokenizing.  This is expensive")
    tokenized_documents = gensim.parsing.preprocessing.preprocess_documents(body_text)
    print("done preprocessing documents. now writing to output to be used in another documents")
    with open("tokenized_documents.pkl", 'wb') as f:
        pickle.dump(tokenized_documents, f)
    print("done writing file")
    
    gc.collect()
    return tokenized_documents
    


# # Look at metadata

# In[ ]:


metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
#metadata.head()
#print(metadata.shape)
metadata.info()


# # collect file paths

# In[ ]:


jsonfiles_pathnames = glob.glob("/kaggle/input/CORD-19-research-challenge/**/*.json", recursive = True)
print(len(jsonfiles_pathnames))


# # Remove stopwords, stem, and tokenize documents

# Stopwords are words common to all text, such as "and" and "the,"  and therefore we remove them.  Stemming reduces words to their root form.  For example, "infected" and "infecting" will both be reduced to "infect."  Tokenization converts the body text of each paper into a vector of whitespace-separated text (words).  These tokens will be treated as semantic features, particularly after stemming.  When we create the document-feature matrix (term frequency corpus) the exact text of the word (or bigram) will no longer be observed, as these features define the row vectors of that matrix.  Here, we save the tokenized documents.

# In[ ]:



tokenized_documents = open_tokenize(jsonfiles_pathnames)


# # Create bigram model

# In this section we create and save the bigram model.  This just contains the word pair.  We will have to run the tokenized documents through the model in order to create a bigram tokenized version of the documents.  We will do that in another step therefore we save the model here.

# In[ ]:


print("creating bigram model.  this is expensive, skipping trigrams")
bigram_model = make_bigram(tokenized_documents)
print("done creating bigram model. Writing model")
with open("bigram_model.pkl", 'wb') as f:
    pickle.dump(bigram_model, f)
print("done writing bigram model")

