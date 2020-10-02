#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Hello world!**
# 
# In this tutorial, I will got through usage of SOTA transformers opensourced by HuggingFace team.
# We will be using BERT transformer model for this tutorial.
# You can check this link to understand more about HuggingFace transformers https://huggingface.co/transformers/pretrained_models.html

# Following are the basic steps involved in using any transformer,
# 
# ### **For preprocessing** 
# 1. Tokenize the input data and other input details such as Attention Mask for BERT to not ignore the attention on padded sequences.
# 2. Convert tokens to input ID sequences.
# 3. Pad the IDs to a fixed length.
# 
# ### **For modelling**
# 1. Load the model and feed in the input ID sequence (Do it batch wise suitably based on the memory available).
# 2. Get the output of the last hidden layer
#     * Last hidden layer has the sequence representation embedding at 0th index, hence we address the output as last_hidden_layer[0].
# 3. These embeddings can be used as the inputs for different machine learning or deep learning models.

# ## **Using BERT Transformer**

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import BertTokenizer, BertModel



# In[ ]:


MODEL_TYPE = 'bert-base-uncased'
MAX_SIZE = 150
BATCH_SIZE = 200


# In[ ]:


train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# ## **Load the required tokenizer and model**

# In[ ]:


tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
model = BertModel.from_pretrained(MODEL_TYPE)


# ## **Convert Text to Tokens**

# In[ ]:


tokenized_input = train_df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[ ]:


print(tokenized_input[1])
print("Here 101 -> [CLS] and 102 -> [SEP]")


# **Here 101 -> [CLS] and 102 -> [SEP]**
# 
# [CLS] token refers to the classification token. We need to take the embedding of the token from the output layer. It represents entire sequence embedding.
# 
# [SEP] refers to end of the sequence.

# Now let's pad the sequence to fixed length

# In[ ]:


padded_tokenized_input = np.array([i + [0]*(MAX_SIZE-len(i)) for i in tokenized_input.values])


# In[ ]:


print(padded_tokenized_input[0])


# Let's tell BERT to ignore attention on padded inputs.

# In[ ]:


attention_masks  = np.where(padded_tokenized_input != 0, 1, 0)


# In[ ]:


print(attention_masks[0])


# In[ ]:


input_ids = torch.tensor(padded_tokenized_input)  
attention_masks = torch.tensor(attention_masks)


# ## Get the sequence embedding

# In[ ]:


all_train_embedding = []

with torch.no_grad():
  for i in tqdm(range(0,len(input_ids),200)):    
    last_hidden_states = model(input_ids[i:min(i+200,len(train_df))], attention_mask = attention_masks[i:min(i+200,len(train_df))])[0][:,0,:].numpy()
    all_train_embedding.append(last_hidden_states)


# In[ ]:


unbatched_train = []
for batch in all_train_embedding:
    for seq in batch:
        unbatched_train.append(seq)

train_labels = train_df['target']


# ##### Now we have the train embeddings.This can be used as an input to other machine learning models

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =  train_test_split(unbatched_train, train_labels, test_size=0.33, random_state=42, stratify=train_labels)


# **References:  **
# 
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# 
# https://huggingface.co/transformers/

# In[ ]:


len(X_train)


# In[ ]:




