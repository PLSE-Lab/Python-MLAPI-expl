#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil
import seaborn as sns

np.set_printoptions(suppress=True)


# **lable selected for our project:** 
# answer_relevance prediction

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


PATH = '/kaggle/input/datanew/'
BERT_PATH = '/kaggle/input/bertbasefromtfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

QUESTION_LABES_VECTOR_SIZE = 21
ANSWER_LABES_VECTOR_SIZE = 4
BOTH_LABES_VECTOR_SIZE = 5

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')

# ##remove later!!!
# df_train = df_train[:10]
# df_test = df_test[:10]
# df_sub = df_sub[:10]

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


# **DATA engineering:**
# 

# 1.NaN handling

# In[ ]:


count = 0
for var in df_train.head(1):
    for elem in df_train[var]:
        if elem == 'NaN' :
            count +=1    
    print(var , count/df_train.shape[0])
    count = 0


# 2.Extraction of 'host' and 'category' to 'sub_category'

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x='category', data=df_train)


# In[ ]:


df_train['host']=df_train['host'].apply(lambda x:(x.replace(x ,x.split('.')[0])))
df_test['host']=df_test['host'].apply(lambda x:(x.replace(x ,x.split('.')[0])))


# In[ ]:


df_train.columns = df_train.columns.str.replace('host', 'sub_category')
df_test.columns = df_test.columns.str.replace('host', 'sub_category')


# In[ ]:


#we trying to figure out if category is redundent due to the sub category
arr_sub_category = {}
for sub in df_train['sub_category'].unique():
    data = df_train[df_train['sub_category'] == sub]
    arr = df_train[df_train['sub_category'] == sub]['category'].unique()
    if len(arr) > 1:
        arr_sub_category[sub] = arr
        print(sub, arr)
        #correlation between two features


# In[ ]:


# split each sub_category found to be multiplied to more than one category 
#(in order to drop the category feature without loosing any important info)
for key in arr_sub_category.keys():
    for val in arr_sub_category[key]:
        df_train[(df_train['sub_category'] == key) & (df_train['category'] == val)] = df_train[(df_train['sub_category'] == key) & (df_train['category'] == val)].apply(lambda x :x.replace(key, key+'_'+val))


# In[ ]:


# checking
print(df_train['sub_category'].unique())


# 3.features plots

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x='sub_category', data=df_train)


# 4.correlation sorted list between features

# In[ ]:


def corrank(X):## to think about relevant correlations (sub_c and lable?, is text coor is relevant?)
       import itertools
       dff = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(method = 'spearman'), 2))],columns=['pairs','corr'])    
       print(dff.sort_values(by='corr',ascending=False))


# In[ ]:


corrank(df_train)


# **5.tokenization?**

# 6.**Categorizing questions**
# #HowTo - instructions (looking for actions in the answer)
# #Why 
# #What / which kind of - classification
# #what is better - comparison
# #when / how often - time
# #what/ how many/ how much - quantative
# #where - places
# #who / whom - person
# #how does / how are - comprehension
# #can/ could - capability
# #should /would you/ do you want / is, are, am / does- Yes/No questions, willing
# #aren't you? wasn't it? - tag questions (in YES/NO)

# In[ ]:


# data_a = np.array([['Can an affidavit be used in Beit Din?', 10], ['How can I write HTML and send as an email?', 15], ['How do I remove a Facebook app request?', 14], ['How do you grapple in Dead Rising 3?', 1], ['How do you make a binary image in Photoshop?', 1]]) 
# df_a = pd.DataFrame(data_a)
# df_a.columns = ["question", "num"]
# print(df_a)

# from adam_qas import adam_script as adam
# dfOut = pd.DataFrame(np.array([["","",""]]))
# dfOut.columns = ["q_class","q_keywords","query"]
# dfOut = dfOut[:-1]
# new_features = adam.activate(df_a['question'], dfOut)
# print(new_features.head(2))
# df_a.join(new_features)
# print(df_a.head(5))


# #### Calling adam's algorithem to produce new features 

# In[ ]:


# import adam_qas as adam
# dfOut = pd.DataFrame(np.array([["","",""]]))
# dfOut.columns = ["q_class","q_keywords","query"]
# dfOut = dfOut[:-1]

# new_features = adam.activate(df_train['question_title'], dfOut)
# print(new_features)

# df_new_train=pd.concat([df_train,new_features], axis=1, sort=False)
# df_new_test=pd.concat([df_test,new_features], axis=1, sort=False)


# In[ ]:


# df_new_train.to_csv(PATH+'df_new_train.csv', index=False)
# df_new_test.to_csv(PATH+'df_new_test.csv', index=False)


# ?.Redefine the features passed to the input categories - add sub_categoty

# In[ ]:


# df_new_train.head(10)


# In[ ]:


# df_new_test.head(10)


# In[ ]:


df_test = pd.read_csv(PATH+'df_new_test.csv', sep=",", encoding="ISO-8859-1")
df_train = pd.read_csv(PATH+'df_new_train.csv', sep=",", encoding="ISO-8859-1")
output_categories_question = list(df_train.columns[11:32])
output_categories_answer = list(df_train.columns[37:41])
output_categories_both = list(df_train.columns[32:37])
input_categories = list(df_train.columns[[1,2,5,10,41,42]]) # we added the sub_categoty!
print('\033[1m' + '\033[91m' + '\033[4m' + 'output categories question:\n\t' +'\033[0m', output_categories_question)
print('\033[1m' + '\033[91m' + '\033[4m' + 'output categories answer:\n\t' +'\033[0m', output_categories_answer)
print('\033[1m' + '\033[91m' + '\033[4m' + 'output categories both:\n\t' +'\033[0m', output_categories_both)
print('\033[1m' + '\033[91m' + '\033[4m' + 'input categories:\n\t' +'\033[0m', input_categories)


# In[ ]:


for col in input_categories:
    max_len = 0
    mean = 0
    min_len = 512
    for elem in df_train[col].unique():
        if len(elem)>max_len:
            max_len = len(elem)
        if mean==0:
            mean = (mean+len(elem))/2
        mean = (mean+len(elem))/2
        if len(elem)<min_len:
            min_len = len(elem)
    print('\033[1m' + '\033[91m' + '\033[4m' + col +":" +'\033[0m')
    print("min: " + str(min_len))
    print("max: " + str(max_len))
    print("mean: " + str(floor(mean)))


# In[ ]:


df_test.head(10)


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

def _trim_input(title, answer, sub_category, q_class, q_keywords, max_sequence_length, 
                t_max_len=150, a_max_len=196, s_c_max_len=25, q_c_max_len=10, q_k_max_len=125):

    t   = tokenizer.tokenize(title)
    a   = tokenizer.tokenize(answer)
    s_c = tokenizer.tokenize(sub_category)
    q_c = tokenizer.tokenize(q_class)
    q_k = tokenizer.tokenize(q_keywords)
    
    t_len   = len(t)
    a_len   = len(a)
    s_c_len = len(s_c)
    q_c_len = len(q_c)
    q_k_len = len(q_k)

  
    if s_c_max_len > s_c_len:
        s_c_new_len = s_c_len
    else:
        s_c_new_len = s_c_max_len

    if q_c_max_len > q_c_len:
        q_c_new_len = q_c_len
    else:
        q_c_new_len = q_c_max_len

    if q_k_max_len > q_k_len:
        q_k_new_len = q_k_len
    else:
        q_k_new_len = q_k_max_len

    if t_max_len > t_len:
        t_new_len = t_len
    else:
        t_new_len = t_max_len

    a_new_len = max_sequence_length - (t_new_len+s_c_new_len+q_c_new_len+q_k_new_len+6)
            
    if t_new_len+s_c_new_len+a_new_len+q_c_new_len+q_k_new_len+6 != max_sequence_length:
        raise ValueError("New sequence length should be %d, but is %d" 
                         % (max_sequence_length, (t_new_len+s_c_new_len+a_new_len+q_c_new_len+q_k_new_len+4)))
    
    a   = a[:a_new_len]
    t   = t[:t_new_len]
    s_c = s_c[:s_c_new_len]
    q_c = q_c[:q_c_new_len]
    q_k = q_k[:q_k_new_len]
    
    return t, a, s_c, q_c, q_k

def _trim_input_answer(answer, q_class, q_keywords, max_sequence_length, 
                a_max_len=371, q_c_max_len=10, q_k_max_len=125):

    a   = tokenizer.tokenize(answer)
    q_c = tokenizer.tokenize(q_class)
    q_k = tokenizer.tokenize(q_keywords)
    
    a_len   = len(a)
    q_c_len = len(q_c)
    q_k_len = len(q_k)

    if q_c_max_len > q_c_len:
        q_c_new_len = q_c_len
    else:
        q_c_new_len = q_c_max_len

    if q_k_max_len > q_k_len:
        q_k_new_len = q_k_len
    else:
        q_k_new_len = q_k_max_len

    a_new_len = max_sequence_length - (q_c_new_len+q_k_new_len+6)
            
    if a_new_len+q_c_new_len+q_k_new_len+6 != max_sequence_length:
        raise ValueError("New sequence length should be %d, but is %d" 
                         % (max_sequence_length, (a_new_len+q_c_new_len+q_k_new_len+4)))
    
    a   = a[:a_new_len]
    q_c = q_c[:q_c_new_len]
    q_k = q_k[:q_k_new_len]
    
    return a, q_c, q_k

def _trim_input_question(title, question, sub_category, q_class, q_keywords, max_sequence_length, 
                t_max_len=150, q_max_len=196, s_c_max_len=25, q_c_max_len=10, q_k_max_len=125):

    t   = tokenizer.tokenize(title)
    q   = tokenizer.tokenize(question)
    s_c = tokenizer.tokenize(sub_category)
    q_c = tokenizer.tokenize(q_class)
    q_k = tokenizer.tokenize(q_keywords)
    
    t_len   = len(t)
    q_len   = len(q)
    s_c_len = len(s_c)
    q_c_len = len(q_c)
    q_k_len = len(q_k)

  
    if s_c_max_len > s_c_len:
        s_c_new_len = s_c_len
    else:
        s_c_new_len = s_c_max_len

    if q_c_max_len > q_c_len:
        q_c_new_len = q_c_len
    else:
        q_c_new_len = q_c_max_len

    if q_k_max_len > q_k_len:
        q_k_new_len = q_k_len
    else:
        q_k_new_len = q_k_max_len

    if t_max_len > t_len:
        t_new_len = t_len
    else:
        t_new_len = t_max_len

    q_new_len = max_sequence_length - (t_new_len+s_c_new_len+q_c_new_len+q_k_new_len+6)
            
    if t_new_len+s_c_new_len+q_new_len+q_c_new_len+q_k_new_len+6 != max_sequence_length:
        raise ValueError("New sequence length should be %d, but is %d" 
                         % (max_sequence_length, (t_new_len+s_c_new_len+q_new_len+q_c_new_len+q_k_new_len+4)))
    
    q   = q[:q_new_len]
    t   = t[:t_new_len]
    s_c = s_c[:s_c_new_len]
    q_c = q_c[:q_c_new_len]
    q_k = q_k[:q_k_new_len]
    
    return t, q, s_c, q_c, q_k

def _convert_to_bert_inputs(title, question, answer, sub_category, q_class, q_keywords, tokenizer, max_sequence_length, input_type):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    if(input_type == 0): #question input
        stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + sub_category + ["[SEP]"] + q_class + ["[SEP]"] + q_keywords + ["[SEP]"]
    if(input_type == 1): #answer inputs
        stoken = ["[CLS]"] + answer + ["[SEP]"] + sub_category + ["[SEP]"] + q_class + ["[SEP]"]
    if(input_type == 2): #both inputs
        stoken = ["[CLS]"] + title + ["[SEP]"] + answer + ["[SEP]"] + sub_category + ["[SEP]"] + q_class + ["[SEP]"] + q_keywords + ["[SEP]"]
    
    
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length, input_type):
    input_ids, input_masks, input_segments = [], [], []

    if(input_type == 0): #question input
        for _, instance in tqdm(df[columns].iterrows()):
            t, q, a, s_c, q_c, q_k = instance.question_title, instance.question_body, instance.answer, instance.sub_category, instance.q_class, instance.q_keywords
            t, q, s_c, q_c, q_k = _trim_input_question(t, q, s_c, q_c, q_k, max_sequence_length)

            ids, masks, segments = _convert_to_bert_inputs(t, q, a, s_c, q_c, q_k, tokenizer, max_sequence_length, 0)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
    if(input_type == 1): #answer inputs
        for _, instance in tqdm(df[columns].iterrows()):
            t, q, a, s_c, q_c, q_k = instance.question_title, instance.question_body, instance.answer, instance.sub_category, instance.q_class, instance.q_keywords
            a, s_c, q_c, = _trim_input_answer(a, s_c, q_c, max_sequence_length)

            ids, masks, segments = _convert_to_bert_inputs(t, q, a, s_c, q_c, q_k, tokenizer, max_sequence_length, 1)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
    if(input_type == 2): #both inputs
        for _, instance in tqdm(df[columns].iterrows()):
            t, q, a, s_c, q_c, q_k = instance.question_title, instance.question_body, instance.answer, instance.sub_category, instance.q_class, instance.q_keywords
            t, a, s_c, q_c, q_k = _trim_input(t, a, s_c, q_c, q_k, max_sequence_length)

            ids, masks, segments = _convert_to_bert_inputs(t, q, a, s_c, q_c, q_k, tokenizer, max_sequence_length, 2)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


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

def bert_model(size):
    
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
    out = tf.keras.layers.Dense(size, activation="sigmoid", name="dense_output")(x)

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


# In[ ]:


gkf = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body) ############## originaln_splits=10

outputs_question = compute_output_arrays(df_train, output_categories_question)
outputs_answer = compute_output_arrays(df_train, output_categories_answer)
outputs_both = compute_output_arrays(df_train, output_categories_both)
inputs_question = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 0) #0 for question inputs
inputs_answer = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 1) #1 for answer inputs
inputs_both = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 2) #2 for both inputs
test_inputs_question = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 0) #0 for question inputs
test_inputs_answer = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 1) #1 for answer inputs
test_inputs_both = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH, 2) #2 for both inputs


# In[ ]:


# histories = []
# for fold, (train_idx, valid_idx) in enumerate(gkf):
    
#     # will actually only do 3 folds (out of 5) to manage < 2h
#     if fold < 3:
#         K.clear_session()
#         model = bert_model(QUESTION_LABES_VECTOR_SIZE)

#         train_inputs = [inputs_question[i][train_idx] for i in range(3)]
#         train_outputs = outputs_question[train_idx]

#         valid_inputs = [inputs_question[i][valid_idx] for i in range(3)]
#         valid_outputs = outputs_question[valid_idx]

#         # history contains two lists of valid and test preds respectively:
#         #  [valid_predictions_{fold}, test_predictions_{fold}]
#         history = train_and_predict(model, 
#                           train_data=(train_inputs, train_outputs), 
#                           valid_data=(valid_inputs, valid_outputs),
#                           test_data=test_inputs_question, 
#                           learning_rate=3e-5, epochs=5, batch_size=8,
#                           loss_function='binary_crossentropy', fold=fold)

#         histories.append(history)


# In[ ]:


# test_predictions = [histories[i].test_predictions for i in range(len(histories))]
# test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
# test_predictions = np.mean(test_predictions, axis=0)

# df_sub.iloc[:, 1:22] = test_predictions

# df_sub.to_csv('submission_question.csv', index=False)


# In[ ]:


histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < 3:
        K.clear_session()
        model = bert_model(ANSWER_LABES_VECTOR_SIZE)

        train_inputs = [inputs_answer[i][train_idx] for i in range(3)]
        train_outputs = outputs_answer[train_idx]

        valid_inputs = [inputs_answer[i][valid_idx] for i in range(3)]
        valid_outputs = outputs_answer[valid_idx]

        # history contains two lists of valid and test preds respectively:
        #  [valid_predictions_{fold}, test_predictions_{fold}]
        history = train_and_predict(model, 
                          train_data=(train_inputs, train_outputs), 
                          valid_data=(valid_inputs, valid_outputs),
                          test_data=test_inputs_answer, 
                          learning_rate=3e-5, epochs=5, batch_size=8,
                          loss_function='binary_crossentropy', fold=fold)

        histories.append(history)


# In[ ]:


test_predictions = [histories[i].test_predictions for i in range(len(histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 27:] = test_predictions

df_sub.to_csv('submission_answer.csv', index=False)


# In[ ]:


# histories = []
# for fold, (train_idx, valid_idx) in enumerate(gkf):
    
#     # will actually only do 3 folds (out of 5) to manage < 2h
#     if fold < 3:
#         K.clear_session()
#         model = bert_model(BOTH_LABES_VECTOR_SIZE)

#         train_inputs = [inputs_both[i][train_idx] for i in range(3)]
#         train_outputs = outputs_both[train_idx]

#         valid_inputs = [inputs_both[i][valid_idx] for i in range(3)]
#         valid_outputs = outputs_both[valid_idx]

#         # history contains two lists of valid and test preds respectively:
#         #  [valid_predictions_{fold}, test_predictions_{fold}]
#         history = train_and_predict(model, 
#                           train_data=(train_inputs, train_outputs), 
#                           valid_data=(valid_inputs, valid_outputs),
#                           test_data=test_inputs_both, 
#                           learning_rate=3e-5, epochs=5, batch_size=8,
#                           loss_function='binary_crossentropy', fold=fold)

#         histories.append(history)


# In[ ]:


# test_predictions = [histories[i].test_predictions for i in range(len(histories))]
# test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
# test_predictions = np.mean(test_predictions, axis=0)

# df_sub.iloc[:, 22:27] = test_predictions

# df_sub.to_csv('submission_both.csv', index=False)


# In[ ]:




