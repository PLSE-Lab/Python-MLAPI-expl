# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:57:13 2019

@author: Amir
"""

#texts = ["This first text talks about houses and dogs",
#        "This is about airplanes and airlines",
#        "This is about dogs and houses too, but also about trees",
#        "Trees and dogs are main characters in this story",
#        "This story is about batman and superman fighting each other", 
#        "Nothing better than another story talking about airplanes, airlines and birds",
#        "Superman defeats batman in the last round"]


#glove_embedding = WordEmbeddings('glove')
#
#sentence = Sentence('animal')

# just embed a sentence using the StackedEmbedding as you would with any single embedding.
#glove_embedding.embed(sentence)

# now check out the embedded tokens.
#for token in sentence:
#    print(token)
#    print(token.embedding)

# make one tensor of all word embeddings of a sentence
#sentence_tensor4 = torch.cat([token.embedding.unsqueeze(0) for token in sentence], dim=0)
#
#cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#output = cos(sentence_tensor4, sentence_tensor6)

from flair.embeddings import WordEmbeddings, CharacterEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings, Sentence
import torch
from tqdm import tqdm 
## tracks progress of loop ##
#Taken Random Column 1 Names - 5 clusters

#array([3, 3, 3, 1, 2, 2, 3, 5, 2, 2, 4, 1, 3, 3], dtype=int32)
column_names = ['Name',
'Location',
'Year',
'Kilometers Driven',
'Fuel Type',
'Transmission',
'Owner Type',
'Mileage',
'Engine',
'Power',
'Seats',
'Distance Travelled','Place','Date']


#Taken Random Column 2 Names - 4 clusters
#array([2, 1, 4, 1, 1, 3], dtype=int32)
column_names2 = ['Qualification',
'Experience',
'Rating',
'Place',
'Profile',
'Miscellaneous Information']


#Taken Random Column 3 Names - 6 clusters
#array([1, 2, 6, 1, 1, 5, 4, 3], dtype=int32)

column_names3 = ['TITLE',
'RESTAURANT ID',
'CUISINES',
'TIME',
'CITY', 
'LOCALITY', 
'RATING', 
'VOTES'] 

### Initialising embeddings (un-comment to use others) ###
glove_embedding = WordEmbeddings('glove')
#character_embeddings = CharacterEmbeddings()
#flair_forward  = FlairEmbeddings('news-forward-fast')
#flair_backward = FlairEmbeddings('news-backward-fast')
#bert_embedding = BertEmbeddings()
#elmo_embedding = ELMoEmbeddings()

#
#stacked_embeddings = StackedEmbeddings( embeddings = [ glove_embedding,
#                                                       flair_forward, 
#                                                       flair_backward,
#                                                       bert_embedding,
#                                                   elmo_embedding
#                                                      ])

# create a sentence #
sentence = Sentence('Analytics Vidhya blogs are Awesome .')
# embed words in sentence #
glove_embedding.embed(sentence)
for token in sentence:
  print(token.embedding)
# data type and size of embedding #
print(type(token.embedding))
# storing size (length) #
z = token.embedding.size()[0]


# creating a tensor for storing sentence embeddings #
s = torch.zeros(0,z)

# iterating Sentence (tqdm tracks progress) #
for tweet in tqdm(column_names):   
  # empty tensor for words #
  w = torch.zeros(0,z)   
  sentence = Sentence(tweet)
  glove_embedding.embed(sentence)
  # for every word #
  for token in sentence:
    # storing word Embeddings of each word in a sentence #
    w = torch.cat((w,token.embedding.view(-1,z)),0)
  # storing sentence Embeddings (mean of embeddings of all words)   #
  s = torch.cat((s, w.mean(dim = 0).view(-1, z)),0)


## tensor to numpy array ##
X = s.numpy()   

from scipy.cluster import  hierarchy
threshold = 0.6
Z = hierarchy.linkage(X,"average", metric="cosine")
C = hierarchy.fcluster(Z, threshold, criterion="distance")
print(C)

