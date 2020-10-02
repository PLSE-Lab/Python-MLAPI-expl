#!/usr/bin/env python
# coding: utf-8

# **This tutorial is based on my previous one which was building a word embedding more a GRU layer.**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd 
import random
# data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import TweetTokenizer,sent_tokenize, word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from sklearn import metrics
import os
import torch
import warnings 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, add
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks,Sequential
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam


import gensim 
from gensim.models import Word2Vec


# **Process to prepare the data:**

# In[3]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
#train=train[:250000]
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')


# In[4]:


identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
for col in identity_columns + ['target']:
    train[col] = np.where(train[col] >= 0.5, True, False)
    


# In[5]:


#Split train in train and validate
train_df, valid_df = train_test_split(train, test_size=0.33, stratify=train['target'])
test_df=test

train_df.loc[:,'size_comment']=train_df.comment_text.apply(lambda x:len(x))
valid_df.loc[:,'size_comment']=valid_df.comment_text.apply(lambda x:len(x))
test_df.loc[:,'size_comment']=test_df.comment_text.apply(lambda x:len(x))

#train_df=train_df[:250000]
train_df=train_df
train_df.loc[:,'set_']="train"
valid_df.loc[:,'set_']="valid"
test_df.loc[:,'set_']="test"


#Set_indices=train_df.loc[:,'set_'][:250000]
Set_indices=train_df.loc[:,'set_']
Set_indices=Set_indices.append(valid_df.loc[:,'set_'])
Set_indices=Set_indices.append(test_df.loc[:,'set_'])


#y_train = train_df['target'][:250000]
y_train = train_df['target']
y_valid = valid_df['target']

#Set_indices_labels=train_df.loc[:,'set_'][:250000]
Set_indices_labels=train_df.loc[:,'set_']
Set_indices_labels=Set_indices_labels.append(valid_df.loc[:,'set_'])


# In[6]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    from tensorflow import set_random_seed
    set_random_seed(2)

seed_everything()


# In[7]:


texts=train_df['comment_text']
texts=texts.append(valid_df['comment_text'])
texts=texts.append(test_df['comment_text'])


print(texts.shape)

labels=train_df['target']
labels=labels.append(valid_df['target'])

print(labels.shape)


# **Tokenization:**

# In[8]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 200
max_words = 50000
embedding_size=100
lr = 1e-3
lr_d = 0


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


# In[9]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[10]:


x_train = data[Set_indices == "train"]
x_val = data[Set_indices == "valid"]
x_test = data[Set_indices == "test"]


y_train = labels[Set_indices_labels == "train"]
y_val = labels[Set_indices_labels == "valid"]

print('Shape of train tensor:', x_train.shape)
print('Shape of validate tensor:', x_val.shape)
print('Shape of test tensor:', x_val.shape)


# **Parsing the GloVe word-embeddings file**

# In[11]:


glove_dir = '../input/glove6b100dtxt'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))



for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# **Building the word embedding matrix**

# In[12]:


embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# **Building the model:**

# In[20]:


import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam

sequence_input = L.Input(shape=(maxlen,), dtype='int32')
embedding_layer = L.Embedding(max_words,embedding_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=True)
x = embedding_layer(sequence_input)
x = L.GRU(embedding_dim)(x)
#x = L.Dense(32, activation='relu')(x)
#x = Model(inputs=sequence_input, outputs=x)

y = L.Input(shape=(1,), dtype='float32')
#y = L.Dense(1, activation='relu')(sequence_input_length)
#y = Model(inputs=sequence_input_length, outputs=y)

combined = concatenate([x, y])

combined = L.Dense(32, activation='relu')(combined)
preds = L.Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[sequence_input, y], outputs=preds)

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

from keras.callbacks import EarlyStopping, ModelCheckpoint

ckpt = ModelCheckpoint(f'gru.hdf5', save_best_only = True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# In[21]:


lr = 0.005
lr_d = 0

history = model.fit([x_train, train_df.size_comment], y_train,
                    epochs=100,
                    batch_size=12000,
                    validation_data=([x_val, valid_df.size_comment], y_val),
                    callbacks = [es,ckpt])


# Look at the loss and the gain in accuracy for each epoch

# In[ ]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# **Run final model with the right number of iteration**

# In[ ]:


# evaluate the model
#loss, accuracy = model.evaluate(x_train, y_train, verbose=2)
#print('Accuracy train: %f' % (accuracy*100))

# evaluate the model
#loss, accuracy = model.evaluate(x_val, y_val, verbose=2)
#print('Accuracy validate: %f' % (accuracy*100))


# **Test my word embedding on my local test with Jigsaw metric**

# In[22]:



pred_val = model.predict([x_val, valid_df.size_comment], batch_size = 12000, verbose = 0)


# In[23]:


MODEL_NAME = 'my_model'
TOXICITY_COLUMN = 'target'
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
valid_df[MODEL_NAME] = pred_val

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

bias_metrics_df = compute_bias_metrics_for_model(valid_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
print(bias_metrics_df)

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    
local_valid=get_final_metric(bias_metrics_df, calculate_overall_auc(valid_df, MODEL_NAME))
print(local_valid)
local_valid.tofile('local_valid.csv',sep=',',format='%10.5f')
#accuracy.tofile('accuracyEmbedding.csv',sep=',',format='%10.5f')


# **Apply the model on the test**

# In[ ]:



pred = model.predict([x_test, test_df.size_comment], batch_size = 12000, verbose = 1)

           
sub = pd.DataFrame({"id": test['id'].values})
sub["prediction"] = pred

sub.to_csv('submission.csv', index=False)

