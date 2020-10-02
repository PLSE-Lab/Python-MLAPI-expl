#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import gc
import sys

from tensorflow import keras
from tensorflow.keras import layers

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


base_path = '/kaggle/input/tensorflow2-question-answering/'
train_file_name = 'simplified-nq-train.jsonl'
test_file_name = 'simplified-nq-test.jsonl'


# ## Read Data In

# In[ ]:


def read_data(file_name, num_records = sys.maxsize): # = sys.maxsize
    current_record = 1
    records = []
    
    with open(os.path.join(base_path, file_name)) as file:
        line = file.readline()
        while(line):
            records.append(json.loads(line))
            line = file.readline()
            if current_record > num_records:
                break
                
            if current_record % 5000 == 0:
                print(current_record)
                gc.collect()
                
            current_record = current_record + 1
    df = pd.DataFrame(records)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'max_records = 10000\ndf_train = read_data(train_file_name, max_records)\ngc.collect()')


# In[ ]:


df_train.head()


# In[ ]:


df_train['question_text'][0]


# In[ ]:


df_train['document_text'][0:3]


# In[ ]:


df_train['long_answer_candidates'][0][54]


# In[ ]:


df_train['document_text'][0]


# In[ ]:


df_train['annotations'][0]


# In[ ]:


df_train['yes_no_answer'] = [item[0]['yes_no_answer'] for item in df_train['annotations']]
df_train['long_answer'] = [item[0]['long_answer'] for item in df_train['annotations']]
df_train['short_answers'] = [item[0]['short_answers'] for item in df_train['annotations']]
df_train['annotation_id'] = [item[0]['annotation_id'] for item in df_train['annotations']]


# In[ ]:


df_train['yes_no_answer'].value_counts()


# ## Getting values out of annotations

# In[ ]:


#Short answer
start_vals = []
end_vals = []

for item in df_train['short_answers']:
    start = -1
    end = -1
    if len(item) > 0:
        start = item[0]['start_token']
        end = item[0]['end_token']
    #if len(item) > 1: #TODO -> there are cases with more than one correct long/short answers, handle/check it
    #    print(item)
    start_vals.append(start)
    end_vals.append(end)
df_train['short_answer_start'] = start_vals
df_train['short_answer_end'] = end_vals

# del df_train['short_answers'] #TODO


# In[ ]:


#Long answer
    
df_train['long_answer_start'] = [item['start_token'] for item in df_train['long_answer']]
df_train['long_answer_end'] = [item['end_token'] for item in df_train['long_answer']]
df_train['long_answer_index'] = [item['candidate_index'] for item in df_train['long_answer']]

# del df_train['long_answer'] #TODO


# In[ ]:


df_train.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.head()


# In[ ]:


df_train = df_train[['document_text', 'question_text', 'short_answer_start', 'short_answer_end', 'annotation_id']]


# In[ ]:


df_train.head(1)


# In[ ]:


df_train['document_text'][0].split()[1955:1969]


# In[ ]:


## HOW TO MODEL THE PROBLEM/architectur
#TRAIN =>
#[..., 'example', 'of', 'permission', 'marketing', 'is', 'a', 'newsletter', 'sent', 'to', 'an', 'advertising', 'firm', "'s", 'customers', ...]
#[..., 0,          0,        0,            0,        0,   1,         1,        1,     1,    1,        1,          1,     1,      1      , ...]


# In[ ]:


# TEST -> possible problem (s) !

# What to do with this scenario:
# [..., 'example', 'of', 'permission', 'marketing', 'is', 'a', 'newsletter', 'sent', 'to', 'an', 'advertising', 'firm', "'s", 'customers', ...]
# [..., 1,          1,        1,            0,        0,   1,         1,        1,     1,    0,        0,          1,     1,      1      , ...]


# In[ ]:


df_train['document_text'][0].split()


# ## MODEL architecture

# In[ ]:


#inputs = [['wikipedia', 'dcoument', 'text', '...'], ['question', 'text', '...']]
inputs = [['wikipedia', 'dcoument', 'text', '...']]
outputs = [0,       1, 1, '...']


# In[ ]:


# ### This code works for individual samples, now we will try to focus on having the same for the entire dataset

# doc_text = df_train['document_text'][0]
# short_answer_start = df_train['short_answer_start'][0]
# short_answer_end = df_train['short_answer_end'][0]

# que_text = df_train['question_text'][0]

# from keras.preprocessing.sequence import pad_sequences

# max_len_document = 2500
# max_len_que = 50
# max_len_word = 40
# max_len_input = max_len_document + max_len_que

# doc_text_lst = []
# doc_text_lst.append(doc_text.split())

# que_text_lst = []
# que_text_lst.append(que_text.split())

# output_label = np.zeros(max_len_input)

# doc_text_lst = pad_sequences(doc_text_lst, maxlen=max_len_document, dtype=object, padding='post', truncating='post', value='')

# que_text_lst = pad_sequences(que_text_lst, maxlen=max_len_que, dtype=object, padding='post', truncating='post', value='')

# if short_answer_end <= max_len_document:
#     output_label[short_answer_start:short_answer_end] = np.ones(short_answer_end - short_answer_start)
    
    
# from keras.preprocessing import text
# def train_tokenizer(train_data):
#     tokenizer = text.Tokenizer(num_words=50, filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n', lower=True, char_level=True) #split='', 
#     tokenizer.fit_on_texts(train_data)
#     return tokenizer

# tokenizer = train_tokenizer(doc_text_lst[0])
# doc_text_lst[0] = tokenizer.texts_to_sequences(doc_text_lst[0])
# doc_text_chars = pad_sequences(doc_text_lst[0], maxlen=max_len_word, padding='post', truncating='post', value=0)

# que_text_lst[0] = tokenizer.texts_to_sequences(que_text_lst[0])
# que_text_chars = pad_sequences(que_text_lst[0], maxlen=max_len_word, padding='post', truncating='post', value=0)


# x_train = np.array(doc_text_chars)

# lst = []


# val_temp = [[item] for item in doc_text_chars]
# _ = [val_temp.append([item]) for item in que_text_chars]

# lst.append(val_temp)
# lst.append(val_temp)
# x_train = np.asarray(lst)

# y_train = []
# y_train.append([[item] for item in output_label])
# y_train.append([[item] for item in output_label])
# y_train = np.asarray(y_train)

# # history = model.fit(x_train, y_train, epochs=5)


# In[ ]:


# for all the samples -> continue from here

doc_text = df_train['document_text']
short_answer_start = df_train['short_answer_start']
short_answer_end = df_train['short_answer_end']
que_text = df_train['question_text']


# In[ ]:



max_len_document = 1000
max_len_que = 50
max_len_word = 10
max_len_input = max_len_document + max_len_que
max_len_input = 1000 ##### TEMP


# In[ ]:




# for item in 
output_labels = []
for index in short_answer_end.index:
    output_label = np.zeros(max_len_input)
     
    end = short_answer_end.iloc[index]
    start = short_answer_start.iloc[index]
        
    if end > -1 and end <= max_len_document:
        output_label[start:end] = np.ones(end - start)
    output_labels.append(output_label)


# In[ ]:





# In[ ]:


from keras.preprocessing import text
def train_tokenizer(train_data):
    tokenizer = text.Tokenizer(num_words=50, filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n', lower=True, char_level=True) #split='', 
    tokenizer.fit_on_texts(train_data)
    return tokenizer

tokenizer = train_tokenizer(doc_text[0:min(800, doc_text.shape[0])])


# In[ ]:


doc_text.shape[0]


# In[ ]:


# from keras.preprocessing.sequence import pad_sequences

# max_len_document = 2500
# max_len_que = 50
# max_len_word = 40
# max_len_input = max_len_document + max_len_que

# doc_text_lst = []

# for item in doc_text:
#     tmp = item.split()
#     doc_text_lst.append(tokenizer.texts_to_sequences(tmp))

# que_text_lst = []

# for item in que_text:
#     tmp = item.split()
#     que_text_lst.append(tokenizer.texts_to_sequences(tmp))
    
# doc_text_lst = pad_sequences(doc_text_lst, maxlen=max_len_document, dtype=object, padding='post', truncating='post', value='')
# que_text_lst = pad_sequences(que_text_lst, maxlen=max_len_que, dtype=object, padding='post', truncating='post', value='')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom keras.preprocessing.sequence import pad_sequences\n\ndoc_text_lst = []\n\nfor item in doc_text:\n    doc_text_lst.append(item.split())\n\nque_text_lst = []\n\nfor item in que_text:\n    que_text_lst.append(item.split())\n    \ndoc_text_lst = pad_sequences(doc_text_lst, maxlen=max_len_document, dtype=object, padding='post', truncating='post', value='')\nque_text_lst = pad_sequences(que_text_lst, maxlen=max_len_que, dtype=object, padding='post', truncating='post', value='')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndoc_text_chars = []\nfor i in range(doc_text_lst.shape[0]):\n    tmp = tokenizer.texts_to_sequences(doc_text_lst[i])\n    doc_text_chars.append(pad_sequences(tmp, maxlen=max_len_word, padding='post', truncating='post', value=0))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "que_text_chars = []\nfor i in range(que_text_lst.shape[0]):\n    tmp = tokenizer.texts_to_sequences(que_text_lst[i])\n    que_text_chars.append(pad_sequences(tmp, maxlen=max_len_word, padding='post', truncating='post', value=0))")


# In[ ]:


doc_text_lst = None
que_text_lst = None


# In[ ]:


x_train = np.array(doc_text_chars)
# TODO -> look into this part later

# lst = []

# val_temp = [[item] for item in doc_text_chars]
# _ = [val_temp.append([item]) for item in que_text_chars]

# lst.append(val_temp)
# lst.append(val_temp)
# x_train = np.asarray(lst)


# In[ ]:


y_train = []
y_train = [[[item2] for item2 in item] for item in output_labels]
y_train = np.asarray(y_train)

# history = model.fit(x_train, y_train, epochs=5)


# Let's join question with document text to make it a single inout to the model

# In[ ]:


y_train.shape


# In[ ]:


x_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom keras import losses\ndoc_input = keras.Input(shape=(max_len_input, max_len_word), name='doc_text')  #TODO -> make the length of the sequences variable\n\nbody_features = doc_input\n\n# body_features = layers.Reshape((max_len_input, 40))(body_features)\n\n#Embed each character in the text into a 64-dimensional vector\nbody_features = layers.Embedding(50, 10)(body_features)\n\nbody_features = layers.TimeDistributed(layers.LSTM(25))(body_features)\n\nshort_answer = layers.TimeDistributed(layers.Dense(1, activation='sigmoid', name='short_answer'))(body_features)\n# Instantiate an end-to-end model predicting both priority and department\nmodel = keras.Model(inputs=doc_input, outputs=short_answer, name='qa_model')\nmodel.compile(loss= losses.binary_crossentropy\n, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])\nmodel.summary()\n\nhistory = model.fit(x_train, y_train, epochs=5, validation_split = 0.2)\n\n# model.summary()")


# In[ ]:


# test_scores = model.evaluate(x_test, y_test, verbose=2)
# print('Test loss:', test_scores[0])
# print('Test accuracy:', test_scores[1])


# ### Covert the outout into submission file

# In[ ]:


x_test = x_train # for now
test_scores = model.predict(x_test, verbose=2)


# In[ ]:


test_scores


# In[ ]:


for i in test_scores[0]:
    if i > 0.5:
        print(i[0])


# In[ ]:


def get_short_answer(single_output):
    answer_start = -1
    answer_end = -1
    i = 0
    for item in single_output:
        if item[0] > 0.5:
            if answer_start == -1:
                answer_start = i 
                answer_end = i
            else:
                answer_end = i
        elif answer_start != -1 :
            break
        i = i + 1
    return answer_start, answer_end


# In[ ]:


test_scores.shape


# In[ ]:


for item in test_scores:
    answer_start, answer_end = get_short_answer(item)
    print(answer_start, answer_end)


# In[ ]:


short_answers = []
long_answers = []
example_id = []
for annotation_id in df_test['annotation_id']:
    example_id.append('-' + str(annotation_id) + '_short')
    example_id.append('-' + str(annotation_id) + '_long')


# In[ ]:


-7853356005143141653_short,YES

