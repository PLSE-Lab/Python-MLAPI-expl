#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')


# # Importing dependencies

# In[ ]:


import tensorflow as tf
import keras
import transformers 
from transformers import * 
import pandas as pd
import numpy as np
import tokenizers


# # Printing tf version
# 

# In[ ]:


print(tf.__version__)


# In[ ]:


vocab_file = '../input/tf-roberta/vocab-roberta-base.json'
merge_file = '../input/tf-roberta/merges-roberta-base.txt'
tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file, merge_file,lowercase = True)


# In[ ]:


MAX_LEN = 100
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
train.head()


# # Tokenization <br/>
# The tokenization logic was inpired from Abhishek Takur

# In[ ]:


ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1) 
        
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids)+5] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1


# In[ ]:


test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


# # Building the model

# In[ ]:


ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
config = RobertaConfig.from_pretrained('../input/tf-roberta/config-roberta-base.json')
bert_model = TFRobertaModel.from_pretrained('../input/tf-roberta/pretrained-roberta-base.h5',config=config)
x = bert_model(ids,attention_mask=att,token_type_ids=tok)
drop1 = tf.keras.layers.Dropout(0.2)(x[0])
layer2 = tf.keras.layers.Conv1D(50 ,kernel_size = 1)(drop1)
drop2 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Conv1D(1 ,kernel_size = 1)(drop2)
layer4 = tf.keras.layers.Flatten()(layer3)
output_1 = tf.keras.layers.Activation('softmax')(layer4)

drop1_ = tf.keras.layers.Dropout(0.2)(x[0])
layer1_ = tf.keras.layers.Conv1D(50 ,kernel_size = 1)(drop1_)
drop2_ = tf.keras.layers.Dropout(0.2)(layer1_)
layer2_ = tf.keras.layers.Conv1D(1 ,kernel_size = 1)(drop2_)
layer3_ = tf.keras.layers.Flatten()(layer2_)
output_2 = tf.keras.layers.Activation('softmax')(layer3_)
model = tf.keras.Model(inputs = [ids ,att ,tok] ,outputs = [output_1 ,output_2])
model.summary()


# In[ ]:


def my_loss(alpha ,gamma):
    '''defining focal loss with gamma and alpha parameters'''
    def focal_loss(y_true ,y_pred):
        y_true = tf.cast(y_true ,dtype = tf.float32)
        y_pred = tf.cast(y_pred ,dtype = tf.float32)
        log_lik = y_true*tf.keras.backend.log(y_pred)
        log_lik = alpha*((1-y_pred)**gamma)*log_lik
        return -tf.keras.backend.sum(log_lik)
    return focal_loss


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss = my_loss(1.0 ,2.0) ,optimizer = optimizer)


# In[ ]:


history = model.fit([input_ids[800:], attention_mask[800:],token_type_ids[800:]], [start_tokens[800:], end_tokens[800:]] ,epochs = 30 ,
         validation_split = 0.1 ,batch_size = 32)


# ### Defining metric
# 

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
oof_start,oof_end = model.predict([input_ids[:800],attention_mask[:800],token_type_ids[:800]],verbose=1)


# In[ ]:


all = []
jac = []
for k in range(800):
    a = np.argmax(oof_start[k,])
    b = np.argmax(oof_end[k,])
    if a>b: 
        st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
    else:
        text1 = " "+" ".join(train.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(jaccard(st,train.loc[k,'selected_text']))
jac.append(np.mean(all))


# In[ ]:


print(np.mean(jac))


# # Kaggle submission
# 

# In[ ]:


preds = model.predict([input_ids_t ,attention_mask_t ,token_type_ids_t] ,verbose = 1)
preds_start = preds[0]
preds_end = preds[1]


# In[ ]:


all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax([preds_start[k ,]])
    b = np.argmax(preds_end[k ,])
    if a>b:
        st = test.loc[k ,'text']
    else:
        text1 = " " + " ".join(test.loc[k ,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)


# In[ ]:


test['selected_text'] = all
test[['textID' ,'selected_text']].to_csv('submission.csv' ,index = False)
pd.set_option('max_colwidth' ,60)
test.sample(25)


# In[ ]:




