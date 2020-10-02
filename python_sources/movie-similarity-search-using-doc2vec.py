#!/usr/bin/env python
# coding: utf-8

# # Model Development for Custom Movie Similarity Search
# This notebook shows how you can take a database of movie descriptions and titles and train an AI model to determine the similiarites of movies by using vectors. The idea is that if we turn each movie into a vector we can determine similar movies and actors by taking the distance from each movie to other movies in the model.

# ## Import libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import multiprocessing as mp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
base_url = '/kaggle/input'
import os
for dirname, _, filenames in os.walk(base_url):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Read in movie database
# Here we are taking the movie database and turning it into a pandas dataframe in order to use the data.

# In[ ]:


df = pd.read_csv(os.path.join(dirname, 'movies_metadata.csv'))
df.head(5)


# ## Turn movie descriptions and titles into a TaggedDocument
# Here we are going through each movie in the database and turning each one into a TaggedDocument object. The TaggedDocument object is a list of words = movie description, tag = title of movie.

# In[ ]:


tagged_data = [TaggedDocument(words=word_tokenize(str(_r['overview']).lower()), tags=[str(_r['title'])]) for i, _r in df.iterrows()]


# In[ ]:


tagged_data[0]


# ## Model Training
# * Define the training criteria and acceptance. (learning rate, max # of epochs (just in case we don't hit our acceptance criteria))
# * Create a Doc2Vec object.
# * Iterate through each TaggedDocument and train the model for each document. Decreasing the learning rate each time.

# In[ ]:


# get the number of cores and that will be the number of workers in the training
num_workers = mp.cpu_count()
print("number of cpus: " + str(num_workers))
# max number of epochs
max_epochs = 1
vec_size = 20
#initial learning rate
alpha = 0.025
#Learning rate will linearly drop to min_alpha as training progresses.
min_alpha = 0.00025

# define the model class.
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=min_alpha,
                min_count=1,
                dm =1,
                workers=num_workers)

# build the vocabulary from the movie description data.
model.build_vocab(tagged_data)

# # start training
# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=max_epochs)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha


# In[ ]:


model_dir = base_url.join("models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "d2v____v1.model"))
print("Model Saved")


# In[ ]:


from gensim.models.doc2vec import Doc2Vec

#model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('Jumanji')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])

