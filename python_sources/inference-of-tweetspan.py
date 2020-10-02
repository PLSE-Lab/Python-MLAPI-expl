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

import os, gc, glob
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
from transformers import *
import tensorflow.keras.layers as L
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import KFold


# In[ ]:


train=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sub=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")


# In[ ]:


BATCH_SIZE = 32 


# In[ ]:


#train["text"]=train["text"].apply(lambda x: str(x).strip())
test["text"]=test["text"].apply(lambda x: str(x).strip())


# In[ ]:


#print("MAX_TRAIN_LEN:",train["text"].str.len().max())
print("MAX_TESTLEN:",test["text"].str.len().max())


# In[ ]:


MAX_LEN = 170


# In[ ]:


def extract_span(df):
    idx = [(full_text.index(sub_text),len(sub_text)) for full_text, sub_text in zip(df["text"].astype(str).values, df["selected_text"].astype(str).values)]
    s_idx = np.array(idx)[:, 0]
    e_idx = np.array(idx)[:, 1]+s_idx
    df["start"] = s_idx
    df["end"] = e_idx


# In[ ]:


extract_span(train)


# In[ ]:


def make_question(df):
    sentiments = df["sentiment"].values
    df["question"] = "What part is "+sentiments
    


# In[ ]:


#make_question(train)
make_question(test)


# In[ ]:


#train_ins = train[["question", "text"]].astype(str).values
test_ins = test[["question", "text"]].astype(str).values


# In[ ]:


#mapped_tr = tuple(map(tuple, train_ins))
mapped_test = tuple(map(tuple, test_ins))


# In[ ]:


#mapped_tr=list(mapped_tr)
mapped_test = list(mapped_test)


# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


MODEL_DIR =  "/kaggle/input/bertlatgetweetqa/"


# In[ ]:


MODEL_TYPE = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR+"tokenizer")
#base_model = TFBertModel.from_pretrained(MODEL_TYPE)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#tr_tokenized = tokenizer.batch_encode_plus(mapped_tr, max_length=MAX_LEN, pad_to_max_length=True)\nts_tokenized = tokenizer.batch_encode_plus(mapped_test, max_length=MAX_LEN, pad_to_max_length=True)')


# In[ ]:


#tr_tokenized.keys()


# In[ ]:


#train["token_ids"] = tr_tokenized["input_ids"]
#train["token_type"] = tr_tokenized["token_type_ids"]
#train["token_mask"] = tr_tokenized["attention_mask"]

test["token_ids"] = ts_tokenized["input_ids"]
test["token_type"] = ts_tokenized["token_type_ids"]
test["token_mask"] = ts_tokenized["attention_mask"]


# In[ ]:


"""config = base_model.config
config.output_hidden_states=True
base_model = TFBertModel(config)"""


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_cell_magic('time', '', 'size=10000\n\ntest_ids = tf.convert_to_tensor(test["token_ids"])\ntest_attn = tf.convert_to_tensor(test["token_mask"])\ntest_id_type = tf.convert_to_tensor(test["token_type"])')


# In[ ]:


#BATCH_SIZE=32


# In[ ]:



test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(((test_ids, test_id_type, test_attn)))
    .batch(BATCH_SIZE)
    #.prefetch(AUTO)
)


# In[ ]:


def loss_fn(y_true, y_pred):

    st_loss = tf.losses.sparse_categorical_crossentropy(y_true[0], tf.squeeze(y_pred[0]), from_logits=True)
    end_loss = tf.losses.sparse_categorical_crossentropy(y_true[1], tf.squeeze(y_pred[1]), from_logits=True)

    return st_loss+end_loss


# In[ ]:


pre_model = TFBertForQuestionAnswering.from_pretrained(MODEL_DIR+"model/")
model_ = pre_model
model_.compile(loss=loss_fn, optimizer="adam")


# In[ ]:


gc.collect()


# In[ ]:


ps = model_.predict((test_ids, test_id_type, test_attn), verbose=1)


# In[ ]:


test_st = ps[0]
test_end = ps[1]


# In[ ]:


ts_s=tf.argmax(tf.nn.softmax(test_st), 1)
ts_end  = tf.argmax(tf.nn.softmax(test_end), 1)


# In[ ]:


ts_s = ts_s.numpy()
ts_end = ts_end.numpy()


# In[ ]:


#average ans span
(ts_end-ts_s).mean()  


# In[ ]:


ts_end<ts_s


# In[ ]:


test["selected_text"] = ""
for i in range(len(test)):
    if(ts_s[i]<ts_end[i]):
        test.loc[i, "selected_text"] = test.loc[i, "text"][ts_s[i]:ts_end[i]]
    else:
        test.loc[i, "selected_text"] = test.loc[i, "text"]


# In[ ]:


sub_df = test[["textID", "selected_text"]]


# In[ ]:


sub_df


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




