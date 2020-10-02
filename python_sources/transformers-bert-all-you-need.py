#!/usr/bin/env python
# coding: utf-8

# To DO
# This notebook is still in progress. 
# * Pre-processing of text
# * Currenty, only using *text* column. Encode other information such as keywords, location etc.

# In[ ]:


import numpy as np # linear algebra
from tqdm import tqdm_notebook
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from tqdm import tqdm_notebook


# In[ ]:


# !pip install transformers


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')


# This notebook approach is derived and inspired from http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

# ![image.png](attachment:image.png)
# 
# Source: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

# 

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# Prepare dataset
# Since Kaggle only allow 16 GB RAM and it would not be enough for processing all training data through it. We need to process the train dataset in few batches 
# 

# After tokenization, tokenized is a list of sentences -- each sentences is represented as a list of tokens. We want BERT to process our examples all at once (as one batch). It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, rather than a list of lists (of different lengths).

# If we directly send padded to BERT, that would slightly confuse it. We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. That's what attention_mask is:

# The model() function runs our sentences through BERT. The results of the processing will be returned into last_hidden_states.

# ## Feature Extraction using DistilBERT

# In[ ]:



def get_features(data, batch_size=2500):
    # Use DistilBERT as feature extractor:
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
     model.to(device)
    
    # tokenize,padding and masking
    tokenized = data["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)
    
    last_hidden_states=[]
    no_batch = data.shape[0]//batch_size
    start_index=0
    end_index=1
    for i in tqdm_notebook(range(1,no_batch+2)):

        if  data.shape[0]>batch_size*i:
                end_index=batch_size*i
        else:
            end_index=train.shape[0]

        input_ids = torch.tensor(padded[start_index:end_index])  
        batch_attention_mask = torch.tensor(attention_mask[start_index:end_index])

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            batch_hidden_state = model(input_ids, attention_mask=batch_attention_mask)
#             import pdb
#             pdb.set_trace()
            print("Batch {} is completed sucessfully".format(i))
            last_hidden_states.append(batch_hidden_state[0])

        start_index=batch_size*i
        end_index=batch_size*i
    fin_features = torch.cat(last_hidden_states,0)
    clf_features = fin_features[:,0,:].numpy()
    return clf_features


# In[ ]:


gc.collect()
features = get_features(train,batch_size=2500)
test_features = get_features(test,batch_size=2500)


# Let's slice only the part of the output that we need. That is the output corresponding the first token of each sentence. The way BERT does sentence classification, is that it adds a token called [CLS] (for classification) at the beginning of every sentence. The output corresponding to that token can be thought of as an embedding for the entire sentence.

# ## Training a Logistic Regression model using features from DistilBERT

# In[ ]:


## Use features from previous modle and train a Logistic regression model
labels = train["target"]
# train model
lr_clf = LogisticRegression()
lr_clf.fit(features, labels)


# In[ ]:


# train_features, test_features, train_labels, test_labels = train_test_split(clf_features, labels,random_state=420)


# ## Create a submission file
# 

# In[ ]:


lr_clf = LogisticRegression()
lr_clf.fit(features, labels)


# In[ ]:


test_pred = lr_clf.predict(test_features)


# In[ ]:



submission['target'] = test_pred
submission.to_csv('submission.csv', index=False)


# In[ ]:




