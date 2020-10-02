#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# from transformers import *
# import tokenizers
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub


# In[ ]:


get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')
import tokenization


# In[ ]:


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for i, text in enumerate(texts):
        
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        if i%100 == 0:
            print(i,"/", len(texts))
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


def build_model(bert_layer, max_len=512):
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     # instantiate a distribution strategy
#     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

#     # instantiating the model in the strategy scope creates the model on the TPU
#     with tpu_strategy.scope():

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


train =pd.read_csv('../input/fake-news/train.csv')
train.info()
train.fillna('null', inplace = True)


# In[ ]:


test = pd.read_csv('../input/fake-news/test.csv')
test.info()
test.fillna('null', inplace=True)


# In[ ]:


train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.label.values


# In[ ]:


np.save('train_input.npy', train_input)
np.save('test_input.npy', test_input)


# In[ ]:


# a = bert_encode(train.text.values[:1], tokenizer, max_len=160)
# a = (tf.convert_to_tensor(a[0], tf.int32), tf.convert_to_tensor(a[1], tf.int32), tf.convert_to_tensor(a[2], tf.int32))


# In[ ]:


model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


train_history = model.fit(
    [train_input[0], train_input[1], train_input[2]], train_labels,
    validation_split=0.2,
    epochs=2,
    batch_size=8,
    verbose = 2
)

model.save('model.h5')


# In[ ]:


a = test
a['label'] = model.predict([test_input[0], test_input[1], test_input[2]])
a['label'] = a['label'].apply(round)
a.head()
a[['id', 'label']].to_csv('submission1.csv',index=False)


# In[ ]:


# # a = bert_encode(train.text.values[:2], tokenizer, max_len=160)
# max_len = 10
# input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
# input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
# segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

# _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
# clf_output = sequence_output[:, 0, :]
# out = Dense(1, activation='sigmoid')(clf_output)
# model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

# model(a)


# PATH = '../input/tf-roberta/'
# tokenizer = tokenizers.ByteLevelBPETokenizer(
#     vocab_file=PATH+'vocab-roberta-base.json', 
#     merges_file=PATH+'merges-roberta-base.txt',
#     add_prefix_space=True,
#     lowercase=True
# )

# m = df.shape[0]
# m

# MAX_LEN = 512
# x_train = np.zeros((m, MAX_LEN), dtype='int32')
# att_tok = np.ones((m, MAX_LEN), dtype='int32')
# tok = np.zeros((m, MAX_LEN), dtype='int32')
# for k in range(m):
#     text = " "+" ".join(df.loc[k,'text'].split())
#     author = " ".join(df.loc[k,'author'].split())
#     title = " "+" ".join(df.loc[k, 'title'].split()) 
# #     embeds = [0] + tokenizer.encode(title).ids + [2, 2] + tokenizer.encode(author).ids + [2, 2] + tokenizer.encode(text).ids + [2]
#     embeds = [0] + tokenizer.encode(text).ids + [2]
#     if len(embeds)<MAX_LEN:
#         temp_len = (MAX_LEN-len(embeds))
#         att_tok[k, len(embeds):] = 0 
#         embeds += temp_len * [0]
#         
#     x_train[k] = embeds[:MAX_LEN]
#     
# y_train = df['label']

# m_test = test.shape[0]

# x_test = np.zeros((m_test, MAX_LEN), dtype='int32')
# att_tok_t = np.ones((m_test, MAX_LEN), dtype='int32')
# tok_t = np.zeros((m_test, MAX_LEN), dtype='int32')
# for k in range(m_test):
#     text = " "+" ".join(test.loc[k,'text'].split())
#     author = " "+" ".join(test.loc[k, 'author'].split())
#     title = " "+" ".join(test.loc[k,'title'].split())
# #     embeds = [0] + tokenizer.encode(title).ids + [2, 2] + tokenizer.encode(author).ids + [2, 2] + tokenizer.encode(text).ids + [2]
#     embeds = [0] + tokenizer.encode(text).ids + [2]
#     if len(embeds)<MAX_LEN:
#         att_tok_t[k, len(embeds):] = 0
#         embeds += (MAX_LEN-len(embeds)) * [0]
#     x_test[k] = embeds[:MAX_LEN]
#     

# def build_model():
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
# 
#     # instantiate a distribution strategy
#     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
#     with tpu_strategy.scope():
#         ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#         att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#         tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
# 
#         config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
#         bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
#         x = bert_model(ids, attention_mask=att, token_type_ids=tok)
# 
#         x1 = tf.keras.layers.Dropout(0.2)(x[0]) 
#         x1 = tf.keras.layers.Conv1D(1,1)(x1)
#         x1 = tf.keras.layers.Flatten()(x1)
#         x1 = tf.keras.layers.Dense(2, activation = 'softmax')(x1)
# 
#         model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1])
#         optimizer = tf.keras.optimizers.Adam()
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# 
#     return model

# model = build_model()
# model.fit([x_train[:16000], att_tok[:16000], tok[:16000]], [y_train[:16000]], epochs = 5, batch_size = 100, verbose=1, validation_data=[[x_train[16000:], att_tok[16000:], tok[16000:]], [y_train[16000:]]])

# # config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
# # bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
# # bert_model(x_train, attention_mask=att_tok, token_type_ids=tok)

# y_train[0]

# x_train
# 

# model.predict([x_test[0:10], att_tok_t[0:10], tok_t[0:10]])

# y = df['label']

# y[:10]

# for i, label in enumerate(y):
#     print(i, label)
# 

# y_train = np.ones((m, 2), dtype='int32')
# 
# for i, label in enumerate(y):
#     if label==0:
#         y_train[i, 1] = 0
#     else:
#         y_train[i, 0] = 0

# y_train[0:10]

# In[ ]:




