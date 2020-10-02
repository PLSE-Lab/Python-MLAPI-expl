#!/usr/bin/env python
# coding: utf-8

# **This notebook is used for predicting.**
# 
# I'm grateful to Kaggle for such an amazing platform for learning and practicing. Jigsaw Multilingual Toxic Comment Classification is a great task for me to try my computational linguistic ideas.
# 
# My linguistic hypothesis:
# 1. The toxic distributions may not be similar for different languages. Therefore, my English training corpus only involved 1:1 toxic vs non-toxic comments.
# 2. Transferring any model trained on one language to another language may lead to some overfit or underfit. Therefore, my post processing included weights for different languages and fine tuning according to a list of profanity words.
# 3. Different models may capture different cross-lingual aspects. Therefore, I tried to train many models of the same/different neural network structures and collect the best combination results.
# 4. A toxic comment has at least one toxic word or sentence, while there exists no toxic elements in a non-toxic comment. In accordance with this hypothesis, I randomly generated more corpus for training.
# 
# Summary of my approach:
# 1. Employed Xlm-roberta-large for multilingual tokenizing.
# 2. Defined two NN structures. Pure Text Model (PT): with pure texts as input, and Mixed Model (MX): with multi-inputs of pure texts and relevant language information about language types and whether the texts were translated from other languages.
# 3. Employed pseudo-labelling for more training corpus. When my public score was below 0.9490, I chose texts with predicted labels either larger than 0.8 or smaller then 0.2. When my public score was above 0.9490, I used all pseudo labelled data.
# 4. Generated more corpus for training. I used my best model to predict the toxicities for all sentences of each comment in the test data, chose the one with the highest predicted toxicity as a toxic sentence when the comment was labelled as toxic in my best submission; and chose the ones with predicted toxicities lower than 0.2 as non-toxic sentences when the comment was labelled as non-toxic in my best submission. I finally joined each toxic sentence with 5 different non-toxic sentences to generate a toxic corpus, and randomly joined some non-toxic sentences to generate a non-toxic corpus.
# 5. For my final submission, I used 3 PT models and 2 MX models. 2 PT models were trained on English data, validation and pseudo labelled data, the other PT model and one MX model were trained on my generated multilingual data, and the other MX model was trained on English, translated multilingual, validation, pseudo labelled data. They were all used to predict on the original test data.
# 6. I used Linear Regression to ensemble predictions from the 5 models, smoothed the results by multiplying language weights and  profanity weights. The Linear Regression model and smoothing weights were learned from 80% of my historical best predictions. At last, the results were blended with my historical best by weights of (0.2, 0.8).

# **0. Set up the environment.**

# In[ ]:


import os

import numpy as np
from sklearn import metrics
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# **1. Hyperparameters and useful functions.**

# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE
# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync#16
MAX_LEN = 192

# Load the transformer tokenizer
MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)

#Define encoder.
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])

#Build a pure text model where language information is not considered.
def build_model_PT(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model

#Build a mixed model where language types are also featured.
def build_model_mix(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_lang_tags = Input(shape=(4,), dtype=tf.float32, name="input_lang_tags")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    x = Concatenate()([cls_token, input_lang_tags])
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_word_ids, input_lang_tags], outputs=out)
    
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model

#Function for coding language information.
def lang_embed(lang, tran):
    lang_codes = {'en':'000', 'es':'100', 'fr':'010',
                  'it':'001', 'pt':'110', 'ru':'101',
                  'tr':'011'}
    tran_codes = {'orig':'0', 'tran':'1'}
    vec = lang_codes[lang]+tran_codes[tran]
    vec = [int(v) for v in vec]
    return vec

#Function for cutting off the middle part of long texts.
def text_process(text):
    ws = text.split(' ')
    if(len(ws)>160):
        text = ' '.join(ws[:160]) + ' ' + ' '.join(ws[-32:])
    return text


# **2. Build the original and translated test data.**

# In[ ]:


test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')

test['content'] = test['content'].apply(lambda x: text_process(x))
#test['translated'] = test['translated'].apply(lambda x: text_process(x))
                       
x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)
lang_tag_test = np.array([lang_embed(row['lang'], 'orig') for _, row in test.iterrows()])

#x_tran_test = regular_encode(test.translated.values, tokenizer, maxlen=MAX_LEN)
#lang_tag_tran_test = np.array([lang_embed('en', 'tran') for _, row in test.iterrows()])


# **3. Load models and predict.**

# ***3.1 Predict with pure text models.***

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with strategy.scope():\n    transformer_layer = TFAutoModel.from_pretrained(MODEL)\n    model = build_model_PT(transformer_layer, max_len=MAX_LEN)\n#model.summary()')


# In[ ]:


#Pure text model trained on generated multilingual data, pseudo labelled and validation data predicts on multilingual.
model.load_weights('/kaggle/input/mymodels/mg2mp4.h5')
test['p1'] = model.predict(x_test, verbose=1)
#test.to_csv('sub1.csv', index=False)


# In[ ]:


#Pure text model trained on English data, validation and pseudo labelled data predicts on multilingual.
model.load_weights('/kaggle/input/mymodels/en2mp1.h5')
test['p2'] = model.predict(x_test, verbose=1)
#test.to_csv('sub2.csv', index=False)


# In[ ]:


#Pure text model trained on English data, validation and pseudo labelled data predicts on multilingual.
model.load_weights('/kaggle/input/mymodels/en2mp4.h5')
test['p3'] = model.predict(x_test, verbose=1)
#test.to_csv('sub3.csv', index=False)


# ***3.2 Predict with mixed language models.***

# In[ ]:


#Clear up the memory first.
del model
from keras import backend as K
import gc
K.clear_session()
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with strategy.scope():\n    transformer_layer = TFAutoModel.from_pretrained(MODEL)\n    model = build_model_mix(transformer_layer, max_len=MAX_LEN)\n#model.summary()')


# In[ ]:


#Mixed model trained on generated multilingual test data, original validation and pseudo labelled data predicts on multilingual.
model.load_weights('/kaggle/input/mymodels/mixmoriggen.h5')
test['p4'] = model.predict([x_test, lang_tag_test], verbose=1)
#test.to_csv('sub4.csv', index=False)


# In[ ]:


#Mixed model trained on English, translated multilingual, validation, pseudo labelled and generated data, predicts on multilingual.
model.load_weights('/kaggle/input/mymodels/mixmorigp3.h5')
test['p5'] = model.predict([x_test, lang_tag_test], verbose=1)
#test.to_csv('sub5.csv', index=False)


# **4. Blend and smooth for submission.**

# In[ ]:


mybest = pd.read_csv('/kaggle/input/mybest/sub9523.csv')
#Blend first.
X = np.array([test.p1.values, test.p2.values, test.p3.values, test.p4.values, test.p5.values]).T
y = mybest.toxic.values
X_T, X_val, y_T, y_val = train_test_split(X, y, test_size=0.2)
    
model = LinearRegression()
model.fit(X, y)
    
prds = model.predict(X_val)
score1 = roc_auc_score(y_val.round().astype(int), prds)
score2 = roc_auc_score(prds.round().astype(int), y_val)
print('Validation:', score1, score2)  

prds = model.predict(X)
score1 = roc_auc_score(y.round().astype(int), prds)
score2 = roc_auc_score(prds.round().astype(int), y)
print('Again my best:', score1, score2)
test['prd'] = prds


# In[ ]:


#Then smooth.
p = [1.3,0.6,0.8,0.5,0.6,0.6]
out = []
for _, row in test.iterrows():
    item = [row['id'], row['prd'], row['lang']]
    if(item[2]=='es'):
        if(item[1]<0.7):
            item[1] *= p[0]
    elif(item[2]=='fr'):
        if(item[1]<0.7):
            item[1] *= p[1]
    elif(item[2]=='ru'):
        if(item[1]<0.7):
            item[1] *= p[2]
    elif(item[2]=='it'):
        if(item[1]<0.7):
            item[1] *= p[3]
    elif(item[2]=='tr'):
        if(item[1]<0.7):
            item[1] *= p[4]
    elif(item[2]=='pt'):
        if(item[1]<0.7):
            item[1] *= p[5]

    out.append([item[0], item[1]])

of = pd.DataFrame(out, columns=['id', 'toxic'])
print(of.head())
score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
print('%2.4f\t%2.4f'%(100*score1, 100*score2))


# **5. Fine tune results with profanity list.**

# In[ ]:


dic = {}
oft = open('/kaggle/input/profanity/Profanity.txt', "r", encoding='utf8')
for l in oft:
    ele = l.strip().lower().split(':')
    dic[ele[0]] = ele[1]
oft.close()

les, lit, ltr, lfr, lru, lpt = 1.2, 1.1, 1.3, 1.2, 1.2, 1.3
of['content'] = test['content']
of['tran'] = test['translated']
of['lang'] = test['lang']
out = []
enpros = dic['en'].split(',')

for _, row in of.iterrows():
    if(row['lang']=='es'):
        lmd = les
    elif(row['lang']=='it'):
        lmd = lit
    elif(row['lang']=='tr'):
        lmd = ltr
    elif(row['lang']=='fr'):
        lmd = lfr
    elif(row['lang']=='ru'):
        lmd = lru
    else:
        lmd = lpt

    item = [row['id'], row['toxic']]
    if(item[1]<0.5):
        for w in enpros:
            if(str(row['tran']).lower().find(w)>=0):
                item[1] *= 1.2
                break

        ws = dic[row['lang']].split(',')
        for w in ws:
            if(str(row['content']).lower().find(w)>=0):
                item[1] *= lmd
                break
    out.append(item)

of = pd.DataFrame(out, columns=['id', 'toxic'])
score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
print('%2.4f\t%2.4f'%(100*score1, 100*score2))


# **6. Ensemble with my historical best.**

# In[ ]:


of['toxic'] = mybest.toxic.values*0.8 + of.toxic.values*0.2
score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
print('%2.4f\t%2.4f'%(100*score1, 100*score2))
print(of.head())
of.to_csv('submission.csv', index=False)

