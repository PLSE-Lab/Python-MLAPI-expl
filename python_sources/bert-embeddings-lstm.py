#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
import pickle
import gc


# In[ ]:


os.listdir()


# In[ ]:


BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
INPUT = '../input/jigsaw-bert-preprocessed-input/'
TEXT_COL = 'comment_text'
MAXLEN = 250


# 1. pip install pytorch-pretrained-bert without internet

# In[ ]:


os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')


# In[ ]:


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


# In[ ]:


BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'


# 2. create BERT model and put on GPU

# In[ ]:


def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(BERT_FP)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


# In[ ]:


# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,do_lower_case = CASED)


# In[ ]:


embedding_matrix = get_bert_embed_matrix()


# In[ ]:


train = pd.read_csv(INPUT + 'train_bert-base-uncased_ids.csv').sample(frac = 1.0, random_state = 23)


# In[ ]:


test = pd.read_csv(INPUT + 'test_bert-base-uncased_ids.csv')


# In[ ]:


x_train = np.zeros((train.shape[0],MAXLEN),dtype=np.int)

for i,ids in tqdm(enumerate(list(train[TEXT_COL]))):

    input_ids = [int(i) for i in ids.split()[:MAXLEN]]
    inp_len = len(input_ids)
    x_train[i,:inp_len] = np.array(input_ids)
    
x_test = np.zeros((test.shape[0],MAXLEN),dtype=np.int)

for i,ids in tqdm(enumerate(list(test[TEXT_COL]))):

    input_ids = [int(i) for i in ids.split()[:MAXLEN]]
    inp_len = len(input_ids)
    x_test[i,:inp_len] = np.array(input_ids)
    
with open('temporary.pickle', mode='wb') as f:
    pickle.dump(x_test, f) # use temporary file to reduce memory

del x_test
gc.collect()


# In[ ]:



identity_columns = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']
y_identities = (train[identity_columns] >= 0.5).astype(int).values

# Overall
weights = np.ones((len(train),)) / 4
# Subgroup
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()

y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values


    


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# In[ ]:


def build_model(embedding_matrix, num_aux_targets, loss_weight):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    words = Input(shape=(MAXLEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model


# In[ ]:


def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]


# In[ ]:


from sklearn.model_selection import train_test_split

tr_ind, val_ind = train_test_split(list(range(len(x_train))) ,test_size = 0.05, random_state = 23)


# In[ ]:


import gc
NUM_MODELS = 1

BATCH_SIZE = 512
EPOCHS = 5
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
checkpoint_predictions = []
checkpoint_val_preds = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1],loss_weight)
    for global_epoch in range(EPOCHS):
        model.fit(x_train[tr_ind],[y_train[tr_ind], y_aux_train[tr_ind]],validation_data = (x_train[val_ind],[y_train[val_ind], y_aux_train[val_ind]]),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            callbacks=[
                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
            ]
        )
        with open('temporary.pickle', mode='rb') as f:
            x_test = pickle.load(f) # use temporary file to reduce memory
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        checkpoint_val_preds.append(model.predict(x_train[val_ind], batch_size=2048)[0].flatten())
        del x_test
        gc.collect()
        weights.append(2 ** global_epoch)
    del model
    gc.collect()

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)


# In[ ]:


val_preds = np.average(checkpoint_val_preds, weights=weights, axis=0)


# In[ ]:


from sklearn.metrics import roc_auc_score

def power_mean(x, p=-5):
    return np.power(np.mean(np.power(x, p)),1/p)

def get_s_auc(y_true,y_pred,y_identity):
    mask = y_identity==1
    try:
        s_auc = roc_auc_score(y_true[mask],y_pred[mask])
    except:
        s_auc = 1
    return s_auc

def get_bpsn_auc(y_true,y_pred,y_identity):
    mask = (y_identity==1) & (y_true==0) | (y_identity==0) & (y_true==1)
    try:
        bpsn_auc = roc_auc_score(y_true[mask],y_pred[mask])
    except:
        bpsn_auc = 1
    return bpsn_auc

def get_bspn_auc(y_true,y_pred,y_identity):
    mask = (y_identity==1) & (y_true==1) | (y_identity==0) & (y_true==0)
    try:
        bspn_auc = roc_auc_score(y_true[mask],y_pred[mask])
    except:
        bspn_auc = 1
    return bspn_auc

def get_total_auc(y_true,y_pred,y_identities):

    N = y_identities.shape[1]
    
    saucs = np.array([get_s_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])
    bpsns = np.array([get_bpsn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])
    bspns = np.array([get_bspn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])

    M_s_auc = power_mean(saucs)
    M_bpsns_auc = power_mean(bpsns)
    M_bspns_auc = power_mean(bspns)
    rauc = roc_auc_score(y_true,y_pred)


    total_auc = M_s_auc + M_bpsns_auc + M_bspns_auc + rauc
    total_auc/= 4

    return total_auc


# In[ ]:


y_train[val_ind][:,0].shape, val_preds.shape,y_identities[val_ind].shape


# In[ ]:


get_total_auc(y_train[val_ind][:,0],val_preds,y_identities[val_ind])


# In[ ]:


df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
df_submit.prediction = predictions
df_submit.to_csv('submission.csv', index=False)

