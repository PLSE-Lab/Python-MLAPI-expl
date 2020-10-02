#!/usr/bin/env python
# coding: utf-8

# This kernel is continued from Save BERT fine-tuning model.
# In this kernel, you can load the weights from that kernel and make a prediction.

# In[3]:


import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json
sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
get_ipython().system("cp -r '../input/kerasbert/keras_bert' '/kaggle/working'")
BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
import tokenization  #Actually keras_bert contains tokenization part, here just for convenience


# ## Load raw model

# In[4]:


from keras_bert.keras_bert.bert import get_model
from keras_bert.keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5,decay=0.01)
maxlen = 50
print('begin_build')

config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True,seq_len=maxlen)
model.summary(line_length=120)


# ## Build classification model
# 
# As the Extract layer extracts only the first token where "['CLS']" used to be, we just take the layer and connect to the single neuron output.

# In[5]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
import codecs

sequence_output  = model.layers[-6].output
pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
model3  = Model(inputs=model.input, outputs=pool_output)
model3.compile(loss='binary_crossentropy', optimizer=adam)
model3.summary()


# ## Prepare Data, Training, Predicting
# 
# First the model need train data like [token_input,seg_input,masked input], here we set all segment input to 0 and all masked input to 1.
# 
# Still I am finding a more efficient way to do token-convert-to-ids

# In[6]:


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a)>max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
        longer += 1
      one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
      all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)
    
nb_epochs=1
bsz = 32
dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
print('build tokenizer done')
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
train_df = train_df.sample(frac=0.01,random_state = 42)
#train_df['comment_text'] = train_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)


# In[7]:


# you can load the weights by this line
print('load model')
model3.load_weights('../input/save-bert-fine-tuning-model/bert_weights.h5')


# In[ ]:


#load test data
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
#test_df['comment_text'] = test_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
eval_lines = test_df['comment_text'].values
print(eval_lines.shape)
print('load data done')
token_input2 = convert_lines(eval_lines,maxlen,tokenizer)
seg_input2 = np.zeros((token_input2.shape[0],maxlen))
mask_input2 = np.ones((token_input2.shape[0],maxlen))
print('test data done')
print(token_input2.shape)
print(seg_input2.shape)
print(mask_input2.shape)
hehe = model3.predict([token_input2, seg_input2, mask_input2],verbose=1,batch_size=bsz)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = hehe
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)


# If you feel this kernel useful, please upvote this kernel and [save part kernel](https://www.kaggle.com/hiromoon166/save-bert-fine-tuning-model).
# And, don't forget to upvote [the great kernel](https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming) by Yue Zhang.
