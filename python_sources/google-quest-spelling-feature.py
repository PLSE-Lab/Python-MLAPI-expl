#!/usr/bin/env python
# coding: utf-8

# In this notebook i show one approach for predicting the 'question_type_spelling' label 
# by building a custom feature using L2distance between USE embeddings of
# * a magic sentence and 
# * input 'answer' data

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow_hub as hub


# In[ ]:


PATH = "../input/google-quest-challenge/"
# we are using the latest model file from tfhub repo
USE_PATH = "https://tfhub.dev/google/universal-sentence-encoder-large/5"


# In[ ]:


# reading in the input data
df_train = pd.read_csv(PATH+'train.csv')
print('training data shape \t= ', df_train.shape)

# we are focussing here only on the 'answer' column so filtering only that for processing
#input_categories = list(df_train.columns[[1,2,5]])
input_categories = list(df_train.columns[[5]])
print('input categories \t= ', input_categories)


# We find how many rows match for this feature 'question_type_spelling' in training data

# In[ ]:


abc = df_train['question_type_spelling'] > 0
print('Total rows with positive values for this feature = ',len(df_train[abc]))


# In[ ]:


# loading the model file
embed = hub.load(USE_PATH)


# Compute the USE embedding vector for each of the rows in input 'answer' column

# In[ ]:


if 1:
  embeddings_train = {}
  for text in input_categories:
    print('Generating Embeddings for input category = ', text)
    train_text = df_train[text].str.replace('?', '.').str.replace('!', '.').tolist()
    
    curr_train_emb = []
    batch_size = 4
    ind = 0
    while ind*batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size]).numpy())
        ind += 1
        
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)  


# I define a magic sentence, which can be used to compare with the targets in answer column. 
# The goal here is find the closest rows which match the meaning or context for predicting the maximum labels in 'question_type_spelling'

# In[ ]:


magic_sentence = "how to pronounce English words"
#magic_sentence = "pronounce and speak English word"
#magic_sentence = "how to pronounce and spell English word "
#magic_sentence = "How to pronounce English spelling word"


# In[ ]:


# copying the embedding vector to match the rows in train_df
if 1:
  curr_train_emb = []
  spell_vector = embed([magic_sentence])
  for i in range(0,df_train.shape[0]):
      curr_train_emb.append(spell_vector.numpy())
        
  spell_embeddings_train = {}      
  spell_embeddings_train['spell_embedding'] = np.vstack(curr_train_emb)


# In[ ]:


l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)
cos_dist = lambda x, y: (x*y).sum(axis=1)


# In[ ]:


# computing the l2 distance and cosine distance between the pairs
dist_features_train = np.array([
    l2_dist(embeddings_train['answer_embedding'], spell_embeddings_train['spell_embedding']),
    cos_dist(embeddings_train['answer_embedding'], spell_embeddings_train['spell_embedding']),
]).T

df_comp = pd.DataFrame(np.hstack([dist_features_train]),
                       columns=['L2', 'Cosine'])


# Sorting based on the L2 distance and analyzing the top n rows. During prediction we can use a threshold based on L2distance that gives the best accuracy.

# In[ ]:


lowest_l2_index = df_comp.sort_values('L2').head(12).index
print(lowest_l2_index)


# In[ ]:


for row in lowest_l2_index:
    print(row, round(df_train.loc[row]['question_type_spelling'],3), df_train.loc[row]['question_title'])


# Some Observations:
# * we are able to get 5 of 12 (accuracy of ~40%)
# * the data seems very noisy with inconsisent labelling
# 
# Refer prediction output
# 
# 1. question title for 930 and 5199 are same, but one is +ve example and another -ve
# 2. same disparity for question 2680 and 3579
# 3. row 592 is -ve example, but has +ve labels in training set rows 362 & 4082 given below

# **For reference printing below the 11 rows in training set which had positive values for 'question_type_spelling'**

# In[ ]:


rows = df_train['question_type_spelling'] > 0
df_train[rows]['question_title']

