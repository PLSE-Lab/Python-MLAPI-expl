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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

get_ipython().system('pip install transformers')
get_ipython().system('pip install bert')
get_ipython().system('pip install tensorflow-gpu')


# In[ ]:


max_seq_length = 128  # Your choice here.
import tensorflow_hub as hub
import tensorflow as tf
import bert
import math
import transformers
from sklearn.model_selection import StratifiedKFold


# In[ ]:


train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train.head()


# In[ ]:


train['text']=train['text'].fillna(" ")
test['text']=test['text'].fillna(" ")
train['selected_text']=train['selected_text'].fillna(" ")


# In[ ]:


for i in range(len(train['sentiment'])):
    if train['sentiment'][i]=='positive':
        train['sentiment'][i]=0
    elif train['sentiment'][i]=='neutral':
        train['sentiment'][i]=1
    elif train['sentiment'][i]=='negative':
        train['sentiment'][i]=2
        
for i in range(len(test['sentiment'])):
    if test['sentiment'][i]=='positive':
        test['sentiment'][i]=0
    elif test['sentiment'][i]=='neutral':
        test['sentiment'][i]=1
    elif test['sentiment'][i]=='negative':
        test['sentiment'][i]=2
train.head()


# In[ ]:


train_x = train['text'].tolist()
# train_x = np.array(train_x, dtype=object)[:, np.newaxis]
train_y = train['selected_text'].tolist()

test_x = test['text'].tolist()
# test_x = np.array(test_x, dtype=object)[:, np.newaxis]
# test_y = test['selected_text'].tolist()


# In[ ]:


sentiment=train['sentiment'].tolist()


# In[ ]:


import re
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS


# In[ ]:


for i in range(len(train_x)):
    train_x[i]=re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', ' ',train_x[i])
    train_x[i]=re.sub(r'^https?:\/\/.*[\r\n]*', ' ', train_x[i], flags=re.MULTILINE)


# In[ ]:


for i in range(len(test_x)):
    test_x[i]=re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]',' ',test_x[i])
    test_x[i]=re.sub(r'^https?:\/\/.*[\r\n]*', ' ', test_x[i], flags=re.MULTILINE)


# # Create Start/End Tokens

# In[ ]:


PRE_TRAINED_MODEL_NAME = 'roberta-large'
tokenizer = transformers.RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


def map_example_to_dict(input_ids, attention_masks,type_ids):
  return [
      tf.convert_to_tensor(input_ids),
      tf.convert_to_tensor(attention_masks),
      tf.convert_to_tensor(type_ids)
  ]


# In[ ]:


train_token=[]
test_token=[]
for i in range(len(train_x)):
    train_token.append(tokenizer.encode_plus(train_x[i],pad_to_max_length=True,max_length=max_seq_length,return_token_type_ids=True))
for i in range(len(test_x)):
    test_token.append(tokenizer.encode_plus(test_x[i],pad_to_max_length=True,max_length=max_seq_length,return_token_type_ids=True))
print(train_token[0])


# In[ ]:


input_ids=[]
attention_mask=[]
type_ids=[]
for i in train_token:
    input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))
    attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))
    type_ids.append(tf.reshape(i['token_type_ids'],(-1,max_seq_length)))
train_input=map_example_to_dict(input_ids,attention_mask,type_ids)
print(len(train_input[0]))
#print(len(train_input[0][0]))


# In[ ]:


input_ids=[]
attention_mask=[]
type_ids=[]
for i in test_token:
    input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))
    attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))
    type_ids.append(tf.reshape(i['token_type_ids'],(-1,max_seq_length)))
test_input=map_example_to_dict(input_ids,attention_mask,type_ids)
print(len(test_input[0]))
#print(len(test_input[0][0]))


# In[ ]:


ids = train_input[0]
masks = train_input[1]
token_ids=train_input[2]

ids = tf.reshape(ids, (-1, max_seq_length,))
print("Input ids shape: ", ids.shape)
masks = tf.reshape(masks, (-1, max_seq_length,))
print("Input Masks shape: ", masks.shape)
token_ids = tf.reshape(token_ids, (-1, max_seq_length,))
print("Token Ids shape: ", token_ids.shape)

ids=ids.numpy()
masks = masks.numpy()
token_ids=token_ids.numpy()


# In[ ]:


test_ids = test_input[0]
test_masks = test_input[1]
test_token_ids=test_input[2]

test_ids = tf.reshape(test_ids, (-1, max_seq_length,))
print("Input ids shape: ", test_ids.shape)
test_masks = tf.reshape(test_masks, (-1, max_seq_length,))
print("Input Masks shape: ", test_masks.shape)
test_token_ids = tf.reshape(test_token_ids, (-1, max_seq_length,))
print("Token Ids shape: ", test_token_ids.shape)

test_ids=test_ids.numpy()
test_masks = test_masks.numpy()
test_token_ids=test_token_ids.numpy()


# In[ ]:


ct = train.shape[0]
start_tokens = np.zeros((ct,max_seq_length),dtype='int32')
end_tokens = np.zeros((ct,max_seq_length),dtype='int32')

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
    for t in enc:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    print(offsets[0])
    
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
            
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1


# In[ ]:


VALID_DATA=0.1
valid_size=int(len(ids)*VALID_DATA)

valid_ids=ids[0:valid_size]
ids=ids[valid_size:]

valid_masks=masks[0:valid_size]
masks=masks[valid_size:]

valid_token_ids=token_ids[0:valid_size]
token_ids=token_ids[valid_size:]

valid_start_tokens=start_tokens[0:valid_size]
start_tokens=start_tokens[valid_size:]

valid_end_tokens=end_tokens[0:valid_size]
end_tokens=end_tokens[valid_size:]


# In[ ]:


print(len(valid_ids))
print(len(ids))


# # Create Model

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


bert_model = transformers.TFRobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),loss='categorical_crossentropy')


# In[ ]:


def build_model():
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)
    token_type = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)
    bert_layer = bert_model([input_ids, attention_mask,token_type])[0]

    x1 = tf.keras.layers.Dropout(0.1)(bert_layer) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    x2 = tf.keras.layers.Dropout(0.1)(bert_layer) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.Model(inputs=[input_ids, attention_mask,token_type], outputs=[x1,x2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model


# In[ ]:


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0, patience=3)


# In[ ]:


jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((ids.shape[0],max_seq_length))
oof_end = np.zeros((ids.shape[0],max_seq_length))
preds_start = np.zeros((test_ids.shape[0],max_seq_length))
preds_end = np.zeros((test_ids.shape[0],max_seq_length))


# In[ ]:


model=build_model()
model.fit([ids, masks, token_ids], [start_tokens, end_tokens], 
        epochs=1, batch_size=8, verbose=DISPLAY,
        validation_data=([valid_ids,valid_masks,valid_token_ids], 
        [valid_start_tokens, valid_end_tokens]))
preds = model.predict([test_ids,test_masks,test_token_ids],verbose=DISPLAY)
preds_start += preds[0]
preds_end += preds[1]

# DISPLAY FOLD JACCARD
all = []
for k in range(len(test_ids)):
    a = np.argmax(oof_start[k,])
    b = np.argmax(oof_end[k,])
    if a>b: 
        st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
    else:
        text1 = " "+" ".join(train.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc[a-1:b])
    all.append(jaccard(st,train.loc[k,'selected_text']))
jac.append(np.mean(all))
#print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
print()


# In[ ]:


# skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
# for fold,(idxT,idxV) in enumerate(skf.split(ids,sentiment)):

#     print('#'*25)
#     print('### FOLD %i'%(fold+1))
#     print('#'*25)
    
#     tf.keras.backend.clear_session()
#     model=build_model()
#     sv = tf.keras.callbacks.ModelCheckpoint(
#         '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
#         save_weights_only=True, mode='auto', save_freq='epoch')
        
#     model.fit([ids[idxT,], masks[idxT,], token_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
#         epochs=1, batch_size=8, verbose=DISPLAY,
#         validation_data=([ids[idxV,],masks[idxV,],token_ids[idxV,]], 
#         [start_tokens[idxV,], end_tokens[idxV,]]))
    
#     #print('Loading model...')
#     #model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
#     #print('Predicting OOF...')
#     #oof_start[idxV,],oof_end[idxV,] = model.predict([\ids[idxV,],masks[idxV,],token_ids[idxV,]],verbose=DISPLAY)
    
#     print('Predicting Test...')
#     preds = model.predict([test_ids,test_masks,test_token_ids],verbose=DISPLAY)
#     preds_start += preds[0]/skf.n_splits
#     preds_end += preds[1]/skf.n_splits
    
#     # DISPLAY FOLD JACCARD
#     all = []
#     for k in idxV:
#         a = np.argmax(oof_start[k,])
#         b = np.argmax(oof_end[k,])
#         if a>b: 
#             st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
#         else:
#             text1 = " "+" ".join(train.loc[k,'text'].split())
#             enc = tokenizer.encode(text1)
#             st = tokenizer.decode(enc[a-1:b])
#         all.append(jaccard(st,train.loc[k,'selected_text']))
#     jac.append(np.mean(all))
#     print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
#     print()


# In[ ]:


print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))


# In[ ]:


all = []
for k in range(test_ids.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test.loc[k,'text']
    else:
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc[a-1:b])
    all.append(st)


# In[ ]:


test['selected_text'] = all
test[['textID','selected_text']].to_csv('submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.sample(25)

