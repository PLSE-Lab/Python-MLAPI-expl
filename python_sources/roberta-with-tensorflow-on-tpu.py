#!/usr/bin/env python
# coding: utf-8

# # Running a Keras kernel on TPU for fast training
# 
# Hello everyone, I have been wondering for days how to speed up the training process with Keras. I have been using the Tensorflow Dataset API for a while and have been struggling it to pass multiple inputs and outputs.
# 
# I finally come up with the solution that I share with you. 
# 
# Please let's note that most of this notebook is Kiram's work: https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712
# 
# I only added the TPU part ;) 
# 
# ** If you like this notebook, feel free to upvote it ;) **

# Note that for the inference part, you need to do it on a GPU since per competition rules, we must not enable Internet on a notebook.
# 
# **v3: I noticed that increasing the batch size from 8 to 16 was lowering the performance. Hence I lowered batch size back down at 8.**

# # Load  data and libraries

# In[ ]:


import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *
import tokenizers


# ## Helper functions

# In[ ]:


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


# In[ ]:


def read_train():
    train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train['text'] = train['text'].astype(str)
    train['selected_text'] = train['selected_text'].astype(str)
    return train

def read_test():
    test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test['text'] = test['text'].astype(str)
    return test

def read_submission():
    test = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    return test
    
train_df = read_train()
test_df = read_test()
submission_df = read_submission()


# In[ ]:


def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# # Data preproccesing

# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

MAX_LEN = 96
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
LEARNING_RATE = 3e-5 * strategy.num_replicas_in_sync 
EPOCHS = 5

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file = PATH + 'vocab-roberta-base.json', 
    merges_file = PATH + 'merges-roberta-base.txt', 
    lowercase = True,
    add_prefix_space=True
)

sentiment_id = {'positive': 1313, 
                'negative': 2430, 
                'neutral': 7974}


# In[ ]:


ct = train_df.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train_df.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+ " ".join(train_df.loc[k,'text'].split())
    text2 = " ".join(train_df.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)] = 1
    
    if text1[idx-1] == ' ': 
        chars[idx-1] = 1 
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
        if sm > 0: 
            toks.append(i) 
        
    s_tok = sentiment_id[train_df.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids) + 5] = 1
    
    if len(toks) > 0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1


# In[ ]:


ct = test_df.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test_df.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test_df.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test_df.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


# In[ ]:


def generate_dataset(idxT, idxV):
    
    # Trainset
    trn_input_ids = input_ids[idxT,]
    trn_att_mask = attention_mask[idxT,]
    trn_token_type_ids = token_type_ids[idxT,]
    
    trn_start_tokens = start_tokens[idxT,]
    trn_end_tokens = end_tokens[idxT,]
    
    # Validation set
    val_input_ids = input_ids[idxV,]
    val_att_mask = attention_mask[idxV,]
    val_token_type_ids = token_type_ids[idxV,]
    
    val_start_tokens = start_tokens[idxV,]
    val_end_tokens = end_tokens[idxV,]
    
    # Generating tf.data object
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(({'input_ids':trn_input_ids, 'attention_mask': trn_att_mask, 'token_type_ids': trn_token_type_ids}, 
                             {'start_tokens': trn_start_tokens, 'end_tokens': trn_end_tokens}))
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices(({'input_ids':val_input_ids, 'attention_mask': val_att_mask, 'token_type_ids': val_token_type_ids}, 
                             {'start_tokens': val_start_tokens, 'end_tokens': val_end_tokens}))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )
    
    return trn_input_ids.shape[0]//BATCH_SIZE, train_dataset, valid_dataset


# # Model

# In[ ]:


def scheduler(epoch):
    return LEARNING_RATE * 0.2**epoch


# In[ ]:


def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name='input_ids')
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name='attention_mask')
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name='token_type_ids')

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5', config=config)
    x = bert_model(ids,
                   attention_mask=att,
                   token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(128, 2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2, padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax', name='start_tokens')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax', name='end_tokens')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# # Train
# We will skip this stage and load already trained model

# In[ ]:


n_splits = 5


# In[ ]:


jac = []
VER='v4'
DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
oof_end = np.zeros((input_ids.shape[0], MAX_LEN))

skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777)

for fold, (idxT, idxV) in enumerate(skf.split(input_ids,train_df.sentiment.values)):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)
    
    # Cleaning everything
    K.clear_session()
    tf.tpu.experimental.initialize_tpu_system(tpu)
    
    # Building model
    with strategy.scope():
        model = build_model()
    
    n_steps, trn_dataset, val_dataset = generate_dataset(idxT, idxV)
        
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')
        
    hist = model.fit(trn_dataset, 
                     epochs=EPOCHS, 
                     verbose=DISPLAY, 
                     callbacks=[sv, reduce_lr],
                     validation_data=val_dataset)
    
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = model.predict(val_dataset, verbose=DISPLAY)
    
    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a>b: 
            st = train_df.loc[k,'text']
        else:
            text1 = " "+" ".join(train_df.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        all.append(jaccard(st,train_df.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    print()


# # Inference

# In[ ]:


'''
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
DISPLAY=1
for i in range(5):
    print('#'*25)
    print('### MODEL %i'%(i+1))
    print('#'*25)
    
    K.clear_session()
    model = build_model()
    model.load_weights('../input/model4/v4-roberta-%i.h5'%i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/n_splits
    preds_end += preds[1]/n_splits
'''


# In[ ]:


'''
all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test_df.loc[k,'text']
    else:
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)
'''


# In[ ]:


#test_df['selected_text'] = all
#test_df[['textID','selected_text']].to_csv('submission.csv',index=False)


# In[ ]:



