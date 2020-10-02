#!/usr/bin/env python
# coding: utf-8

# * V14: train with tri grams and generate new vocab, num feat = 15000

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' pip install underthesea')


# In[ ]:


import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import cohen_kappa_score

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import seaborn as sns

from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords

import keras
import io
import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from underthesea import word_tokenize

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import codecs


# In[ ]:


def create_stopwordlist():
    f = codecs.open('/kaggle/input/vietnamese-stopwords/vietnamese-stopwords.txt', encoding='utf-8')
    data = []
    null_data = []
    for i, line in enumerate(f):
        line = repr(line)
        line = line[1:len(line)-3]
        data.append(line)
    return data


# In[ ]:


stopword_vn = create_stopwordlist()


# In[ ]:


import string

def tokenize(text):
    text =  text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in word_tokenize(text.lower()) if word not in stopword_vn]


# In[ ]:


def choose_vectorizer(option, name='tf_idf'):
    if option == 'generate':
        if name == 'tf_idf':
            vectorizer = TfidfVectorizer(tokenizer = tokenize,ngram_range=(1,4), min_df=5, max_df= 0.8, max_features= 5000, sublinear_tf=True)
        else:
            vectorizer = CountVectorizer(tokenizer = tokenize, ngram_range=(1,4), max_df=0.8, min_df=5, max_features = 5000, sublinear_tf=True)
    elif option == 'load':
        if name == 'tf_idf':
            vectorizer = TfidfVectorizer(vocabulary = pickle.load(open('../input/kpdl-data/vocabulary_2.pkl', 'rb')), ngram_range=(1,3), min_df=5, max_df= 0.8, max_features=15000, sublinear_tf=True)
        else:
            vectorizer = CountVectorizer(vocabulary = pickle.load(open('../input/kpdl-data/vocabulary_2.pkl', 'rb')), ngram_range=(1,3), max_df=0.8, min_df=5, max_features = 15000, sublinear_tf=True)
    return vectorizer


# In[ ]:


# data = pd.read_csv('../input/kpdl-data/train_v1.csv')
data = pd.read_csv('../input/vietnamese-sentiment-analyst/data - data.csv')

data.head(2)


# In[ ]:


df = data.loc[data['comment'] == None]


# In[ ]:


df


# In[ ]:


category = data['rate'].unique()
category_to_id = {cate: idx for idx, cate in enumerate(category)}
id_to_category = {idx: cate for idx, cate in enumerate(category)}
print(category_to_id)
print(id_to_category)


# ### Distribution of label

# In[ ]:


data_label = data['rate']
data_label = pd.DataFrame(data_label, columns=['rate']).groupby('rate').size()
data_label.plot.pie(figsize=(15, 15), autopct="%.2f%%", fontsize=12)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['rate'], test_size = .15, shuffle = True, stratify=data['rate'])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = .2, shuffle = True, stratify=y_train)


# In[ ]:


X_train.shape


# In[ ]:


X_train_df = pd.DataFrame(X_train, columns=['comment'])
X_valid_df = pd.DataFrame(X_valid, columns=['comment'])
X_test_df = pd.DataFrame(X_test, columns=['comment'])
print(X_train_df.head(10))
print(X_valid_df.head(10))
print(X_test_df.head(10))


# In[ ]:


y_train


# ### Check distribution of train, valid and test set

# In[ ]:


get_ipython().run_cell_magic('time', '', "options = ['generate', 'load']\n# 0 to generate, 1 to load (choose wisely, your life depends on it!)\noption = options[0] \nvectorizer = choose_vectorizer(option)\n\nX_train = vectorizer.fit_transform(X_train).toarray()\nX_valid = vectorizer.transform(X_valid).toarray()\nX_test = vectorizer.transform(X_test).toarray()\n    \nif option == 'generate':\n    pickle.dump(vectorizer.vocabulary_, open('vocabulary_3.pkl', 'wb'))")


# In[ ]:


print(X_train.shape, X_valid.shape, X_test.shape)


# In[ ]:


X_train.shape


# In[ ]:


y_ = y_train.map(category_to_id).values
y_train = np.zeros((len(y_), y_.max()+1))
y_train[np.arange(len(y_)), y_] = 1
# y_train = y_

y_ = y_test.map(category_to_id).values
y_test = np.zeros((len(y_), y_.max()+1))
y_test[np.arange(len(y_)), y_] = 1
# y_test = y_

y_ = y_valid.map(category_to_id).values
y_valid = np.zeros((len(y_), y_.max()+1))
y_valid[np.arange(len(y_)), y_] = 1
# y_valid = y_


# In[ ]:


print(y_train.sum(1))
print(y_valid.sum(1))
print(y_test.sum(1))
print(y_train.shape, y_valid.shape, y_test.shape)


# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
DROPOUT = 0.3
ACTIVATION = "relu"

model = Sequential([    
    Dense(1000, activation=ACTIVATION, input_dim=X_train.shape[1]),
    Dropout(DROPOUT),
    Dense(500, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(300, activation=ACTIVATION),
    Dropout(DROPOUT),
#     Dense(200, activation=ACTIVATION),
#     Dropout(DROPOUT),
#     Dense(100, activation=ACTIVATION),
#     Dropout(DROPOUT),
#     Dense(50, activation=ACTIVATION),
#     Dropout(DROPOUT),
    Dense(5, activation='softmax'),
])


# In[ ]:


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


model.compile(optimizer=optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc', f1_m,precision_m, recall_m])

model.summary()
es = EarlyStopping(monitor='val_f1_m', mode='max', verbose=1, patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_f1_m', factor=0.2, patience=8, min_lr=1e7)
checkpoint = ModelCheckpoint('best_full.h5', monitor='val_f1_m', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


# In[ ]:


EPOCHS = 25
BATCHSIZE = 4


# In[ ]:


model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(X_valid, y_valid), callbacks=[es, reduce_lr, checkpoint])


# In[ ]:


x = np.arange(EPOCHS)
history = model.history.history


# In[ ]:


# import tensorflow as tf 

# model = tf.keras.models.load_model('../input/kpdl-base/my_model.h5')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

model.save('my_model.h5')


# ### Predict in train, valid and test set

# In[ ]:


predict_train = model.predict(X_train)
predict_valid = model.predict(X_valid)
predict_test = model.predict(X_test)
print(predict_train.shape, predict_valid.shape, predict_test.shape)


# In[ ]:


predict_train_label = predict_train.argmax(-1)
predict_valid_label = predict_valid.argmax(-1)
predict_test_label = predict_test.argmax(-1)


# In[ ]:


# predict_train_label


# In[ ]:


predict_train_label = [id_to_category[predict_train_label[idx]] for idx in range(len(predict_train))]
predict_valid_label = [id_to_category[predict_valid_label[idx]] for idx in range(len(predict_valid))]
predict_test_label = [id_to_category[predict_test_label[idx]] for idx in range(len(predict_test))]


# In[ ]:


# predict_train_label


# In[ ]:


y_train_true = y_train.argmax(-1)
y_valid_true = y_valid.argmax(-1)
y_test_true = y_test.argmax(-1)


# In[ ]:


# y_train_true = y_train
# y_valid_true = y_valid
# y_test_true = y_test


# In[ ]:


# y_train_true


# In[ ]:


# y_train_true


# In[ ]:


y_train_label = [id_to_category[y_train_true[idx]] for idx in range(len(y_train_true))]
y_valid_label = [id_to_category[y_valid_true[idx]] for idx in range(len(y_valid_true))]
y_test_label = [id_to_category[y_test_true[idx]] for idx in range(len(y_test_true))]


# In[ ]:


train_concat = np.concatenate((np.array(X_train_df['comment'].values).reshape(-1, 1), np.array(y_train_label).reshape(-1, 1), np.array(predict_train_label).reshape(-1, 1)), axis=-1)
valid_concat = np.concatenate((np.array(X_valid_df['comment'].values).reshape(-1, 1), np.array(y_valid_label).reshape(-1, 1), np.array(predict_valid_label).reshape(-1, 1)), axis=-1)
test_concat = np.concatenate((np.array(X_test_df['comment'].values).reshape(-1, 1), np.array(y_test_label).reshape(-1, 1), np.array(predict_test_label).reshape(-1, 1)), axis=-1)


# In[ ]:


# train_concat = np.concatenate((np.array(X_train_df['Content'].values).reshape(-1, 1), np.array(y_train_label).reshape(-1, 1)), axis=-1)
# valid_concat = np.concatenate((np.array(X_valid_df['Content'].values).reshape(-1, 1), np.array(y_valid_label).reshape(-1, 1)), axis=-1)
# test_concat = np.concatenate((np.array(X_test_df['Content'].values).reshape(-1, 1), np.array(y_test_label).reshape(-1, 1)), axis=-1)


# In[ ]:


# train_concat_predict_df = pd.DataFrame(train_concat, columns=['Content', 'True_Label'])
# valid_concat_predict_df = pd.DataFrame(valid_concat, columns=['Content', 'True_Label'])
# test_concat_predict_df = pd.DataFrame(test_concat, columns=['Content', 'True_Label'])


# In[ ]:


train_concat_predict_df = pd.DataFrame(train_concat, columns=['comment', 'True_Label', 'Predict'])
valid_concat_predict_df = pd.DataFrame(valid_concat, columns=['comment', 'True_Label', 'Predict'])
test_concat_predict_df = pd.DataFrame(test_concat, columns=['comment', 'True_Label', 'Predict'])


# In[ ]:


train_concat_predict_df.head(20)


# In[ ]:


valid_concat_predict_df.head(20)


# In[ ]:


test_concat_predict_df.head(20)


# ### Save predict to csv file

# In[ ]:


train_concat_predict_df.to_csv('train_concat_predict_df.csv', index=False)
valid_concat_predict_df.to_csv('valid_concat_predict_df.csv', index=False)
test_concat_predict_df.to_csv('test_concat_predict_df.csv', index=False)


# In[ ]:


predict_test


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(8, 8))
conf_mat = confusion_matrix(y_test_true, predict_test.argmax(-1))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=id_to_category.values(), yticklabels=id_to_category.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[ ]:


# y_test_true


# In[ ]:


predict_test = predict_test.argmax(-1)


# In[ ]:


from sklearn.metrics import f1_score
score = f1_score(y_test_true, predict_test, average='weighted')


# In[ ]:


score


# ### List Content that model predicted false

# In[ ]:


labels = data['rate'].unique()


# In[ ]:


for label in labels:
    wrong = []
    df = test_concat_predict_df.loc[test_concat_predict_df['True_Label'] == label]
    df_content = df.values
    for row in df_content:
        if np.abs(int(row[1])- int(row[2])):
            wrong.append(row)
    df_wrong = pd.DataFrame(wrong, columns=['rate', 'true', 'predict'])
    df_wrong.to_csv(f'{label}_test.csv')
    print(label, df_wrong)


# In[ ]:


for label in labels:
    wrong = []
    df = valid_concat_predict_df.loc[valid_concat_predict_df['True_Label'] == label]
    df_content = df.values
    for row in df_content:
        if np.abs(int(row[1])- int(row[2])):
            wrong.append(row)
    df_wrong = pd.DataFrame(wrong, columns=['rate', 'true', 'predict'])
    df_wrong.to_csv(f'{label}_valid.csv')
    print(label, df_wrong.head())


# In[ ]:


for label in labels:
    wrong = []
    df = train_concat_predict_df.loc[train_concat_predict_df['True_Label'] == label]
    df_content = df.values
    for row in df_content:
        if np.abs(int(row[1])- int(row[2])):
            wrong.append(row)
    df_wrong = pd.DataFrame(wrong, columns=['rate', 'true', 'predict'])
    df_wrong.to_csv(f'{label}_train.csv')
    print(label, df_wrong.head())

