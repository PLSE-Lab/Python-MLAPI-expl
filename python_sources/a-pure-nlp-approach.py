#!/usr/bin/env python
# coding: utf-8

# ### Idea
# >Can we predict whether teachers' project proposals are accepted using just NLP? The idea is to build an end-to-end Deep Learning solution to [DonorsChoose.org Application Screening](https://www.kaggle.com/c/donorschoose-application-screening) competition.
# 
# **Note:** This kernel is a work in progress. We will regularly update it with the latest results. Stay tuned and don't forget to upvote!

# In[2]:


# Some essential imports and data loading
import copy
import gc
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv("../input/donorschoose-application-screening/train.csv")
test = pd.read_csv("../input/donorschoose-application-screening/test.csv")
train["set"] = "train"
test["set"] = "test"
data = pd.concat([train, test]).reset_index(drop=True)
assert train.shape[0] + test.shape[0] == data.shape[0]
del train, test
gc.collect();


# ### Preprocessing
# > The first step is to get preprocessed text from the data. We concatenate `project_title`, `project_essay_1`, `project_essay_2`, `project_essay_3`, `project_essay_4`, and `project_resource_summary`

# In[ ]:


def clean_data(data, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]", remove_repetitions=True):
    '''
    preprocess a DataFrame with text
    '''
    data = copy.deepcopy(data)
    
    if lower:
        data = data.str.lower()
        
    if initial_filters is not None:
        data = data.str.replace(initial_filters, ' ')
        
    if remove_repetitions:
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
        data = data.str.replace(pattern, r"\1")
        
    data = data.str.replace(r"[^a-z' ]", ' ')
    return data.str.split().str.join(" ")


# In[ ]:


data = data.fillna(" ")
cols_to_drop = list(set(data.columns) - set(["id", "project_is_approved", "set"]))
data["application_text"] = data["project_title"].str.cat([data["project_essay_%d"%(i+1)] for i in range(4)] + [data["project_resource_summary"]], sep=" ")
data.drop(cols_to_drop, axis=1, inplace=True)
data.application_text = clean_data(data.application_text, 
                                   initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]")


# In[ ]:


data.application_text.apply(lambda x: len(x.split())).hist(bins=100)
plt.show()


# ### Training a Bi-directional GRU
# 
# > Hyperparemeters
# * **`embedding_file`**: FastText 2 million word vectors trained on Common Crawl (600B tokens), i.e., crawl-300d-2M.vec
# *  **`batch_size`**: 256 
# * **`sentence_length`**: 500 (We pad the shorter sentences and slice off longer sentences)
# * **`recurrent_units`**:  64 (For Bidirectional GRU layer)
# * **`dropout_rate`**: 0.3 (For the Dropout layer)
# * **`dense_size`**:  32 (For Dense layer)
# * ** `fold_count`**: 10
# * ** `early_stopping`**: True (patience = 5)

# In[ ]:


'''
def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model
'''


# > ### Submission
# The whole training took more than 6 hours on NVIDIA GeForce GTX 960M. After training, we pickled the predictions in .npy format. It is available here: https://www.kaggle.com/fizzbuzz/donorschooseorg-application-screening-predictions
# 
# We will use these 10-fold predictions and do a Rank-wise average to get our final submission

# In[3]:


from scipy.stats import rankdata

LABELS = ["project_is_approved"]

base = "../input/donorschooseorg-application-screening-predictions/predictions/predictions/"
predict_list = []
for i in range(1,3):
    for j in range(10):
        predict_list.append(np.load(base + "predictions_%03d/test_predicts%d.npy"%(i, j)))
    
print(len(predict_list))
predcitions = np.zeros_like(predict_list[0])
for predict in predict_list:
    predcitions = np.add(predcitions.flatten(), rankdata(predict)/predcitions.shape[0])  
predcitions /= len(predict_list)

submission = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
submission[LABELS] = predcitions
submission.to_csv('submission.csv', index=False)


# In[ ]:




