#!/usr/bin/env python
# coding: utf-8

# ### Bert-base TensorFlow 2.0
# 
# This kernel does not explore the data. For that you could check out some of the great EDA kernels: [introduction](https://www.kaggle.com/corochann/google-quest-first-data-introduction), [getting started](https://www.kaggle.com/phoenix9032/get-started-with-your-questions-eda-model-nn) & [another getting started](https://www.kaggle.com/hamditarek/get-started-with-nlp-lda-lsa). This kernel is an example of a TensorFlow 2.0 Bert-base implementation, using TensorFow Hub. <br><br>
# 
# ---
# **Update 1 (Commit 7):**
# * removing penultimate dense layer; now there's only one dense layer (output layer) for fine-tuning
# * using BERT's sequence_output instead of pooled_output as input for the dense layer
# ---
# 
# **Update 2 (Commit 8):**
# * adjusting `_trim_input()` --- now have a q_max_len and a_max_len, instead of 'keeping the ratio the same' while trimming.
# * **importantly:** now also includes question_title for the input sequence
# ---
# 
# **Update 3 (Commit 9)**
# <br><br>*A lot of experiments can be made with the title + body + answer sequence. Feel free to look into e.g. (1) inventing new tokens (add it to '../input/path-to-bert-folder/assets/vocab.txt'), (2) keeping [SEP] between title and body but modify `_get_segments()`, (3) using the [PAD] token, or (4) merging title and body without any kind of separation. In this commit I'm doing (2). I also tried (3) offline, and they both perform better than in commit 8, in terms of validation rho.*
# * Ignoring first [SEP] token in `_get_segments()`
# ---

# Heres the revised code.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import random
import warnings
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from nltk import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#from ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from googleqa_utilityscript import *
import seaborn as sns
SEED = 0
#seed_everything(SEED)
warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
plt.rcParams.update({'font.size': 16})
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from tqdm import tqdm_notebook, tqdm
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#/kaggle/input/google-quest-challenge/sample_submission.c


# In[ ]:


train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

train['set'] = 'train'
test['set'] = 'test'
complete_set = train.append(test)

print('Train samples: %s' % len(train))
print('Test samples: %s' % len(test))
display(train.head())


# In[ ]:


samp_id = 9
print('Question Title: %s \n' % train['question_title'].values[samp_id])
print('Question Body: %s \n' % train['question_body'].values[samp_id])
print('Answer: %s' % train['answer'].values[samp_id])


# In[ ]:


question_target_cols = ['question_asker_intent_understanding','question_body_critical', 'question_conversational', 
                        'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
                        'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 
                        'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
                        'question_type_compare', 'question_type_consequence', 'question_type_definition', 
                        'question_type_entity', 'question_type_instructions', 'question_type_procedure',
                        'question_type_reason_explanation', 'question_type_spelling', 'question_well_written']
answer_target_cols = ['answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
                      'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 
                      'answer_type_reason_explanation', 'answer_well_written']
target_cols = question_target_cols + answer_target_cols

print('Question labels')
display(train.iloc[[samp_id]][question_target_cols])
print('Answer labels')
display(train.iloc[[samp_id]][answer_target_cols])


# In[ ]:


train_users = set(train['question_user_page'].unique())
test_users = set(test['question_user_page'].unique())

print('Unique users in train set: %s' % len(train_users))
print('Unique users in test set: %s' % len(test_users))
print('Users in both sets: %s' % len(train_users & test_users))
print('What users are in both sets? %s' % list(train_users & test_users))


# In[ ]:


train_users = set(train['answer_user_page'].unique())
test_users = set(test['answer_user_page'].unique())

print('Unique users in train set: %s' % len(train_users))
print('Unique users in test set: %s' % len(test_users))
print('Users in both sets: %s' % len(train_users & test_users))


# In[ ]:


question_gp = complete_set[['qa_id', 'question_user_name', 'question_user_page']].groupby(['question_user_name', 'question_user_page'], as_index=False).count()
question_gp.columns = ['question_user_name', 'question_user_page', 'count']
display(question_gp.sort_values('count', ascending=False).head())

train_question_gp = train[['qa_id', 'question_user_page']].groupby('question_user_page', as_index=False).count()
test_question_gp = test[['qa_id', 'question_user_page']].groupby('question_user_page', as_index=False).count()
train_question_gp.columns = ['question_user_page', 'Question count']
test_question_gp.columns = ['question_user_page', 'Question count']

sns.set(style="darkgrid")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
sns.countplot(x="Question count", data=train_question_gp, palette="Set3", ax=ax1).set_title("Train")
sns.countplot(x="Question count", data=test_question_gp, palette="Set3", ax=ax2).set_title("Test")
plt.show()


# In[ ]:


answer_gp = complete_set[['qa_id', 'answer_user_name', 'answer_user_page']].groupby(['answer_user_name', 'answer_user_page'], as_index=False).count()
answer_gp.columns = ['answer_user_name', 'answer_user_page', 'count']
display(answer_gp.sort_values('count', ascending=False).head())

train_answer_gp = train[['qa_id', 'answer_user_page']].groupby('answer_user_page', as_index=False).count()
test_answer_gp = test[['qa_id', 'answer_user_page']].groupby('answer_user_page', as_index=False).count()
train_answer_gp.columns = ['answer_user_page', 'Answer count']
test_answer_gp.columns = ['answer_user_page', 'Answer count']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
sns.countplot(x="Answer count", data=train_answer_gp, palette="Set3", ax=ax1).set_title("Train")
sns.countplot(x="Answer count", data=test_answer_gp, palette="Set3", ax=ax2).set_title("Test")
plt.show()


# In[ ]:


question_title_gp = complete_set[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()
question_title_gp.columns = ['question_title', 'count']
display(question_title_gp.sort_values('count', ascending=False).head())

train_question_title_gp = train[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()
test_question_title_gp = test[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()
train_question_title_gp.columns = ['question_title', 'Question title count']
test_question_title_gp.columns = ['question_title', 'Question title count']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
sns.countplot(x="Question title count", data=train_question_title_gp, palette="Set3", ax=ax1).set_title("Train")
sns.countplot(x="Question title count", data=test_question_title_gp, palette="Set3", ax=ax2).set_title("Test")
plt.show()


# In[ ]:


question_body_gp = complete_set[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()
question_body_gp.columns = ['question_body', 'count']
display(question_body_gp.sort_values('count', ascending=False).head())

train_question_body_gp = train[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()
test_question_body_gp = test[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()
train_question_body_gp.columns = ['question_body', 'Question body count']
test_question_body_gp.columns = ['question_body', 'Question body count']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
sns.countplot(x="Question body count", data=train_question_body_gp, palette="Set3", ax=ax1).set_title("Train")
sns.countplot(x="Question body count", data=test_question_body_gp, palette="Set3", ax=ax2).set_title("Test")
plt.show()


# In[ ]:


complete_set['question_title_len'] = complete_set['question_title'].apply(lambda x : len(x))
complete_set['question_body_len'] = complete_set['question_body'].apply(lambda x : len(x))
complete_set['answer_len'] = complete_set['answer'].apply(lambda x : len(x))
complete_set['question_title_wordCnt'] = complete_set['question_title'].apply(lambda x : len(x.split(' ')))
complete_set['question_body_wordCnt'] = complete_set['question_body'].apply(lambda x : len(x.split(' ')))
complete_set['answer_wordCnt'] = complete_set['answer'].apply(lambda x : len(x.split(' ')))

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['question_title_len'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['question_title_len'], ax=ax2).set_title("Test")
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['question_title_wordCnt'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['question_title_wordCnt'], ax=ax2).set_title("Test")
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['question_body_len'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['question_body_len'], ax=ax2).set_title("Test")
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['question_body_wordCnt'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['question_body_wordCnt'], ax=ax2).set_title("Test")
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['answer_len'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['answer_len'], ax=ax2).set_title("Test")
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
sns.distplot(complete_set[complete_set['set'] == 'train']['answer_wordCnt'], ax=ax1).set_title("Train")
sns.distplot(complete_set[complete_set['set'] == 'test']['answer_wordCnt'], ax=ax2).set_title("Test")
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7), sharex=True)
sns.countplot(complete_set[complete_set['set'] == 'train']['category'], ax=ax1).set_title("Train")
sns.countplot(complete_set[complete_set['set'] == 'test']['category'], ax=ax2).set_title("Test")
plt.show()


# In[ ]:


complete_set['host_first'] = complete_set['host'].apply(lambda x : x.split('.')[0])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharex=True)
sns.countplot(y=complete_set[complete_set['set'] == 'train']['host_first'], ax=ax1, palette="muted").set_title("Train")
sns.countplot(y=complete_set[complete_set['set'] == 'test']['host_first'], ax=ax2, palette="muted").set_title("Test")
plt.show()


# In[ ]:


f = plt.subplots(figsize=(24, 7))
for col in question_target_cols[:5]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()

f = plt.subplots(figsize=(24, 7))
for col in question_target_cols[5:10]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()

f = plt.subplots(figsize=(24, 7))
for col in question_target_cols[10:15]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()

f = plt.subplots(figsize=(24, 7))
for col in question_target_cols[15:]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()


# In[ ]:


f = plt.subplots(figsize=(24, 7))
for col in answer_target_cols[:5]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()

f = plt.subplots(figsize=(24, 7))
for col in answer_target_cols[5:]:
    sns.distplot(train[col], label=col, rug=True, hist=False)
plt.show()


# In[ ]:


eng_stopwords = stopwords.words('english')

complete_set['question_title'] = complete_set['question_title'].str.replace('[^a-z ]','')
complete_set['question_body'] = complete_set['question_body'].str.replace('[^a-z ]','')
complete_set['answer'] = complete_set['answer'].str.replace('[^a-z ]','')
complete_set['question_title'] = complete_set['question_title'].apply(lambda x: x.lower())
complete_set['question_body'] = complete_set['question_body'].apply(lambda x: x.lower())
complete_set['answer'] = complete_set['answer'].apply(lambda x: x.lower())

freq_dist = FreqDist([word for comment in complete_set['question_title'] for word in comment.split() if word not in eng_stopwords])
plt.figure(figsize=(20, 6))
plt.title('Word frequency on question title').set_fontsize(20)
freq_dist.plot(60, marker='.', markersize=10)
plt.show()

freq_dist = FreqDist([word for comment in complete_set['question_body'] for word in comment.split() if word not in eng_stopwords])
plt.figure(figsize=(20, 6))
plt.title('Word frequency on question body').set_fontsize(20)
freq_dist.plot(60, marker='.', markersize=10)
plt.show()

freq_dist = FreqDist([word for comment in complete_set['answer'] for word in comment.split() if word not in eng_stopwords])
plt.figure(figsize=(20, 6))
plt.title('Word frequency on answer').set_fontsize(20)
freq_dist.plot(60, marker='.', markersize=10)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
from bert_tokenization import tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil

np.set_printoptions(suppress=True)


# In[ ]:


pip install bert


# #### 1. Read data and tokenizer
# 
# Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)

# In[ ]:


PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# #### 2. Preprocessing functions
# 
# These are some functions that will be used to preprocess the raw text data into useable Bert inputs.

# In[ ]:


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# #### 3. Create model
# 
# `compute_spearmanr()` is used to compute the competition metric for the validation set
# <br><br>
# `CustomCallback()` is a class which inherits from `tf.keras.callbacks.Callback` and will compute and append validation score and validation/test predictions respectively, after each epoch.
# <br><br>
# `bert_model()` contains the actual architecture that will be used to finetune BERT to our dataset. It's simple, just taking the sequence_output of the bert_layer and pass it to an AveragePooling layer and finally to an output layer of 30 units (30 classes that we have to predict)
# <br><br>
# `train_and_predict()` this function will be run to train and obtain predictions

# In[ ]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )

def bert_model():
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size,
        fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback


# #### 4. Obtain inputs and targets, as well as the indices of the train/validation splits

# In[ ]:


gkf = GroupKFold(n_splits=5).split(X=df_train.question_body, groups=df_train.question_body)

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# #### 5. Training, validation and testing
# 
# Loops over the folds in gkf and trains each fold for 5 epochs --- with a learning rate of 1e-5 and batch_size of 8. A simple binary crossentropy is used as the objective-/loss-function. 

# In[ ]:



histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < 3:
        K.clear_session()
        model = bert_model()

        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]

        # history contains two lists of valid and test preds respectively:
        #  [valid_predictions_{fold}, test_predictions_{fold}]
        history = train_and_predict(model, 
                          train_data=(train_inputs, train_outputs), 
                          valid_data=(valid_inputs, valid_outputs),
                          test_data=test_inputs, 
                          learning_rate=3e-5, epochs=4, batch_size=8,
                          loss_function='binary_crossentropy', fold=fold)

        histories.append(history)


# #### 6. Process and submit test predictions
# 
# First the test predictions are read from the list of lists of `histories`. Then each test prediction list (in lists) is averaged. Then a mean of the averages is computed to get a single prediction for each data point. Finally, this is saved to `submission.csv`

# In[ ]:


test_predictions = [histories[i].test_predictions for i in range(len(histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 1:] = test_predictions

df_sub.to_csv('submission.csv', index=False)


# In[ ]:


submission.to_csv('submit.csv', index=False)


# In[ ]:


submission


# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
submission = pd.read_csv("../input/abbbbbb/submission.csv")


# In[ ]:




