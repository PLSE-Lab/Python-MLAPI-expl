#!/usr/bin/env python
# coding: utf-8

# ## Intro
# This notebook is a combination of four great notebooks.
# * @xhlulu [Disaster NLP: Keras BERT using TFHub](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub)
# * @Dieter [BERT-Embeddings + LSTM](https://www.kaggle.com/christofhenkel/bert-embeddings-lstm/notebook)
# * @Wojtek Rosa [Keras BERT using TFHub (modified train data)](https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data)
# * @Gunes Evitan [NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert)
# 
# Thanks to their great works. I combine the bert_model from @xhlulu, LSTM model from @Dieter and modified data from @Wojtek Rosa and @Gunes Evitan.

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Bidirectional, SpatialDropout1D, Embedding, add, concatenate
from tensorflow.keras.layers import GRU, GlobalAveragePooling1D, LSTM, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_hub as hub

import tokenization


# In[ ]:


train = pd.read_csv('../input/preprocesseddata/train.csv')
test = pd.read_csv('../input/preprocesseddata/test.csv')
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:


def bert_encoder(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segments_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segments_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


def build_model(bert_layer, max_len=512):
 
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    x = SpatialDropout1D(0.3)(sequence_output)
    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)
    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=result)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


train_input = bert_encoder(train.text.values, tokenizer, max_len=160)
test_input = bert_encoder(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(
                        monitor='val_acc',
                        patience=1,
                        verbose=1,
                        factor=0.5,
                        min_lr=0.001)


# In[ ]:


BATCH_SIZE = 16
LSTM_UNITS = 64
EPOCHS = 5
DENSE_HIDDEN_UNITS = 256

model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


history = model.fit(train_input, train_labels,
                    validation_split=.2,
                    epochs = EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks = [learning_rate_reduction])


# In[ ]:


test_pred = model.predict(test_input)


# In[ ]:


submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)

