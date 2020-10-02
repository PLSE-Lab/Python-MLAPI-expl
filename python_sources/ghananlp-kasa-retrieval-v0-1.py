#!/usr/bin/env python
# coding: utf-8

# # Preliminaries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


# install sent2vec
get_ipython().system('pip install git+https://github.com/epfml/sent2vec')
# install annoy
get_ipython().system('pip install annoy')


# Write requirements to file, anytime you run it, in case you have to go back and recover dependencies.
# 
# Latest known such requirements are hosted for each notebook in the companion github repo, and can be pulled down and installed here if needed. Companion github repo is located at https://github.com/azunre/transfer-learning-for-nlp

# In[ ]:


get_ipython().system('pip freeze > kaggle_image_requirements.txt')


# Check what is in the input folder

# In[ ]:


# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.

get_ipython().system('ls ../input')


# # Load The Raw Data

# In[ ]:


import re
import time

start = time.time()
english_sentences = []
with open("../input/jw300entw/jw300.en-tw.en") as f:
    for line in f:
        english_sentences.append(re.sub(r'[\W\d]', " ",line.lower())) # clean and normalize
end = time.time()
print("Loading the english sentences took %d seconds"%(end-start))


# In[ ]:


print("A sample of the english sentences is:")
print(english_sentences[:10])
print("The length of the list is:")
print(len(english_sentences))


# In[ ]:


twi_sentences = []
with open("../input/jw300entw/jw300.en-tw.tw") as f:
    for line in f:
        twi_sentences.append(re.sub(r'[\W\d]', " ", line.lower())) # clean and normalize


# In[ ]:


print("A sample of the twi sentences is:")
print(twi_sentences[:10])
print("The length of the list is:")
print(len(twi_sentences))


# # Vectorize Subset of English Sentences with sent2vec

# In[ ]:


MAXNSENT = 10000 # how many sentences to take from top of data for now (small experiment)


# In[ ]:


import time
import sent2vec

model = sent2vec.Sent2vecModel()
start=time.time()
model.load_model('../input/sent2vec/wiki_unigrams.bin')
end = time.time()
print("Loading the sent2vec embedding took %d seconds"%(end-start))


# In[ ]:


def assemble_embedding_vectors(data):
    out = None
    for item in data:
        vec = model.embed_sentence(item)
        if vec is not None:
            if out is not None:
                out = np.concatenate((out,vec),axis=0)
            else:
                out = vec                                            
        else:
            pass
        
    return out


# In[ ]:


start=time.time()
EmbeddingVectors = assemble_embedding_vectors(english_sentences[:MAXNSENT])
end = time.time()
print("Computing all embeddings took %d seconds"%(end-start))
print(EmbeddingVectors)


# In[ ]:


print("The shape of embedding matrix:")
print(EmbeddingVectors.shape)


# In[ ]:


# Save embeddings for later use
np.save("english_sent2vec_vectors_jw.npy",EmbeddingVectors)


# # Build and Test Index w/ Annoy for fast Neareast-Neighbor Retrieval
# 
# First build the annoy index for the available English sent2vec vectors

# In[ ]:


from annoy import AnnoyIndex

start = time.time()
dimension = EmbeddingVectors.shape[1] # Length of item vector that will be indexed
english_NN_index = AnnoyIndex(dimension, 'angular')  
for i in range(EmbeddingVectors.shape[0]): # go through every embedding vector
    english_NN_index.add_item(i, EmbeddingVectors[i]) # add to index

english_NN_index.build(10) # 10 trees
english_NN_index.save('en_sent2vec_NN_index.ann') # save index
end = time.time()
print("Building the NN index took %d seconds"%(end-start))


# Test the built index

# In[ ]:


test_english_NN_index = AnnoyIndex(dimension, 'angular')
test_english_NN_index.load('en_sent2vec_NN_index.ann') # super fast, will just mmap the file


# In[ ]:


translation_idx = 5 # choose index of sentence to focus on in english_sentences/twi_sentences

annoy_out = test_english_NN_index.get_nns_by_item(translation_idx, 5) # will 5 nearest neighbors to the very first sentence


# In[ ]:


print(annoy_out)


# In[ ]:


print("- The sentence we are finding nearest neighbors of:\n")
print(english_sentences[annoy_out[0]])
print("\n\n- The 4 nearest neighbors found:\n")
for i in range(1,5):
    print(str(i) + ". "+ english_sentences[annoy_out[i]])


# In[ ]:


print("- In other words, if we were translating the english sentence:\n")
print(english_sentences[annoy_out[0]])
print("  where the known correct translation is:")
print(twi_sentences[annoy_out[0]])
print("\n\n- The 4 top translation suggested by our sparse retrieval system above are:\n")
for i in range(1,5):
    print(str(i) + ". "+ twi_sentences[annoy_out[i]])


# This seems to work! Now we need to just scale it out to the whole dataset and test with random input!
