#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras

get_ipython().system('git clone https://github.com/kpe/bert-for-tf2.git')
get_ipython().system('pip install -r bert-for-tf2/requirements.txt')
get_ipython().system('pip install sentencepiece')

sys.path.append("bert-for-tf2/")

import bert
from bert.model import BertModelLayer
from bert.loader import params_from_pretrained_ckpt, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[ ]:


df_train = pd.read_csv("../input/appletwittersentimenttexts/apple-twitter-sentiment-texts.csv")
df_train.head()

data = df_train['text'].values
labels = df_train['sentiment'].values+1 # if there is -1 in labels, loss could be nan

x_train_text, x_valid_text, y_train, y_valid = train_test_split(data, labels, test_size=0.10, shuffle= True)


# In[ ]:


SEQ_LEN = 128
CLASS = 3
MODEL_PATH = '../input/bert-pretrained-models/multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12/'


# In[ ]:


tokenizer = FullTokenizer(MODEL_PATH + 'vocab.txt', do_lower_case=False)

train_tokens = []
for row in x_train_text:
    train_tokens.append( ["[CLS]"] + tokenizer.tokenize(str(row)) + ["[SEP]"] )

train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
train_token_ids = map(lambda tids: tids + [0] * (SEQ_LEN - len(tids)), train_token_ids)
train_token_ids = np.array([np.array(xi) for xi in list(train_token_ids)])

valid_tokens = []
for row in x_valid_text:
    valid_tokens.append( ["[CLS]"] + tokenizer.tokenize(str(row)) + ["[SEP]"] )

valid_token_ids = list(map(tokenizer.convert_tokens_to_ids, valid_tokens))
valid_token_ids = map(lambda tids: tids + [0] * (SEQ_LEN - len(tids)), valid_token_ids)
valid_token_ids = np.array([np.array(xi) for xi in list(valid_token_ids)])

x_train = train_token_ids
x_valid = valid_token_ids


# In[ ]:


bert_params = params_from_pretrained_ckpt(MODEL_PATH)
bert_layer = BertModelLayer.from_params(bert_params, name="bert")
bert_layer.apply_adapter_freeze()

def create_model(max_seq_length, classes):
    inputs = Input(shape=(max_seq_length,), dtype='int32', name='input_ids')
    bert = bert_layer(inputs)
    cls_out = Lambda(lambda seq: seq[:, 0, :])(bert)
    dr_1 = Dropout(0.3)(cls_out)
    fc_1 = Dense(64, activation=tf.nn.relu)(dr_1)
    dr_2 = Dropout(0.3)(fc_1)
    outputs = Dense(classes, activation='softmax')(dr_2)
    
    model = Model(inputs, outputs)
    
    return model

model = create_model(SEQ_LEN, CLASS)
model.build(input_shape=(None, SEQ_LEN))

load_stock_weights(bert_layer, MODEL_PATH+"bert_model.ckpt")

def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer

for layer in flatten_layers(bert_layer):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        else:
            layer.trainable = False

bert_layer.embeddings_layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])

print(model.summary())


# In[ ]:


checkpointName = "bert_fine-tuning.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointName,
                                                  save_weights_only=True,
                                                  verbose=1)

history = model.fit(x_train, y_train, 
                    epochs=4, batch_size=16,
                    validation_data=(x_valid, y_valid),
                    verbose=1, callbacks=[cp_callback]
)


# In[ ]:




