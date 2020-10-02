#!/usr/bin/env python
# coding: utf-8

# # Task: What do we know about non-pharmaceutical interventions?

# ## Install/Load Packages

# The first block of code is (almost) directly from kaggle

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# There are too many paths and printing them takes
# up too much space so I don't do this normally
if 1==0: 
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            pass
            #print(os.path.join(dirname, filename))


# Next we installl scispacy, a repo of commands to deal with scientific documents. *Note that internet access needs to be switched on for this to work!*

# In[ ]:


# Progress bar
import tqdm

# Word2Vec
from gensim.models.word2vec import Word2Vec

from nltk.tokenize import word_tokenize 
from scipy.spatial.distance import cdist


# ## Introduction

# Now that all our libraries are loaded we need data. We explore the full text in the files using the output generated from the following notebook:
# https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

# In[ ]:


biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")
clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")
clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")
clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")

all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)

all_data.head()


# In[ ]:


print("Number of Rows in Table: %i" % len(all_data))
print("Number of Titles: %i " % all_data['title'].count())
print("Number of Abstracts: %i " % all_data['abstract'].count())
print("Number of Texts: %i " % all_data['text'].count())


# ## Train Word2Vec Model
# The input for the word2vec function is a list of documents where each document is a list of words.

# To get our word2vec model we first get all the text from each document into a list. We do this using our `text` column from `all_data`.

# In[ ]:


# replace newlines with empty strings
all_text = all_data.text.str.replace('\n\n', ' ')
# remove citations
ptn = r'\[[0-9]{1,2}\]'
all_text = all_text.str.replace(ptn,'')


# So `all_text` is now a list of documents. However, instead of the text in each document being a string, we need the texts to be formatted as a list of words that make up the document. (Ideally, we would want real words and not numbers that stand for citations so we probably should clean this next step up eventually, but cest la vie). In other words, We are making a list of documents and then in the list of documents is lists of words in all_text_list. Just so you know the cool computational lingo, each words is also known as a token. 

# In[ ]:


all_text_list = list(
    map(
        lambda x: word_tokenize(x), all_text.values
    ))


# Now we can perform word2vec. Word2vec converts words to vectors using a neural networ.  Using the context of each word, it predicts a vector that represents how it is used contextually. In other words, the vectors created by word2vec are highly dependent on the texts it is trained on. These vectors for words may be very different if trained on scientific journals verses twitter data.

# In[ ]:


model = Word2Vec(all_text_list, size=300, iter=10)


# ## Generate Document Vectors

# Ok so now `model` is like a dictionary that converts words into vectors. For each document, we convert all the words into a vector based on the model. We then make a list of these word vectors and store them in the object `word_vector`. Since each word vector is of the same length, we take the average of all the vectors in each document to get a single vector to represent the document. We set this averaged vector to the document's paper id in the dictionary `document_dict`. 

# In[ ]:


document_dict = {}
for idx, text_df in tqdm.tqdm(all_data[["paper_id", "text"]].iterrows()):
    text = text_df['text'].replace("\n\n", ' ')
    
    word_vector = (
        list(
            map( # this part takes every word and converts it into a vector
                lambda x: model.wv[x], 
                filter( #this part checks that word is in the model
                    lambda x: x in model.wv, 
                    text.split(" ")
                )
            )
        )
    )
    # here we are making a dictory where the document is a key and the value
    # is the average of the word vectors in that document
    if len(word_vector) > 0:
        document_dict[text_df['paper_id']] = pd.np.stack(word_vector, axis=1).mean(axis=1)


# In[ ]:


print(len(document_dict))


# In[ ]:


print(list(document_dict.values())[0].shape)


# Let's convert this dictionary to a dataframe

# In[ ]:


document_embeddings_df = pd.DataFrame.from_dict(document_dict, orient="index")
document_embeddings_df.head()


# ## Document Cosine Similarity

# Now we will use costine similarity to measure how related documents are to the  a specific word. For example of how cosine similarity works see this [image](https://datascience-enthusiast.com/figures/cosine_sim.pn), Higher numbers indicate documents that are closer to the word vector of interest.

# In[ ]:


def cos_sim(text):
    word_vector_list = (
        list(
            map( # this part takes every word and converts it into a vector
                lambda x: model.wv[x], 
                filter( #this part checks that word is in the model
                    lambda x: x in model.wv, 
                    text.split(" ")
                )
            )
        )
    )
    mean_word_vec = pd.np.stack(word_vector_list, axis=1).mean(axis=1)
    

    
    # compute similarity
    doc_sim = (
        1-cdist(
            document_embeddings_df.values,
            [mean_word_vec],
            'cosine'
        )
    )
    # convert result to a date frame
    document_sim_df = (
        pd.DataFrame(doc_sim, columns=["cos_sim"])
        .assign(document_id=list(document_embeddings_df.index))
    )
    # sort from most similar to least
    document_sim_df = document_sim_df.sort_values("cos_sim", ascending=False)
    
    # perform left-join to get information about the documents
    doc_sim_meta_df = document_sim_df.merge(all_data,
                      how='left',
                     left_on='document_id',
                     right_on='paper_id')
    return(doc_sim_meta_df)


# In[ ]:


pd.options.display.max_colwidth = 100
cos_sim('airborne').loc[range(0,10),"title"]


# In[ ]:


pd.options.display.max_colwidth = 100
cos_sim('non-pharmaceutical interventions').loc[range(0,10),"title"]

