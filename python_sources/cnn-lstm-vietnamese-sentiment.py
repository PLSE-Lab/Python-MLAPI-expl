#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(1234)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_log_error
from underthesea import word_tokenize
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.


# In[ ]:


cols = ['text','sentiment']
df_train = pd.read_csv("../input/newone/train.csv",header=None, names=cols)
df_test = pd.read_csv("../input/newone/Test.csv",header=None, names=cols)
df_dev = pd.read_csv("../input/newone/Dev.csv",header=None, names=cols)


# In[ ]:


from keras.utils.np_utils import to_categorical
labels_train = to_categorical(df_train.sentiment, num_classes=3)
labels_test = to_categorical(df_test.sentiment, num_classes=3)
labels_dev = to_categorical(df_dev.sentiment, num_classes=3)


# In[ ]:


df_train['sentiment']= labels_train
df_test['sentiment']= labels_test
df_dev['sentiment']= labels_dev


# In[ ]:


import string
def cleaner_update(text):
    return text.translate(str.maketrans('','', string.punctuation))


# ## Remove punctuation

# In[ ]:


clean_test = []
for i in range(0,len(df_test)):
    clean_test.append(cleaner_update(df_test['text'][i]))


# In[ ]:


clean_dev = []
for i in range(0,len(df_dev)):
    clean_dev.append(cleaner_update(df_dev['text'][i]))


# In[ ]:


clean_train = []
for i in range(0,len(df_train)):
    clean_train.append(cleaner_update(df_train['text'][i]))


# ## Tokenize Words

# In[ ]:


tokenize_df=[]
for x in clean_test:
    tokenize_df.append(word_tokenize(x))


# In[ ]:


for x in clean_train:
    tokenize_df.append(word_tokenize(x))


# In[ ]:


for x in clean_dev:
    tokenize_df.append(word_tokenize(x))


# In[ ]:


tokenize_df


# ### Count Words

# In[ ]:


words=[]
for m in range(0,len(tokenize_df)):
    for n in range(0,len(tokenize_df[m])):
        words.append(tokenize_df[m][n])


# ## Visualize frequency of word (used for stopwords)

# In[ ]:


df_Count = pd.DataFrame(words,columns=['word'])
df_Count['Num']= 1


# In[ ]:


df_GroupBy=df_Count.groupby('word').count()
df_GroupBy.sort_values('Num',ascending=False,inplace=True)


# In[ ]:


filename = '../input/stopword/StopWord.csv'
data = pd.read_csv(filename,names=['word'])
list_stopwords = data['word']
myarray = np.asarray(list_stopwords)
def remove_stopword(text):
    text2=''
    for x in text:
        if x in myarray:
            text2+=""
        else:
            text2+=x+ " "
    return text2
storage=[]
for x in range(0,len(tokenize_df)):
    storage.append(remove_stopword(tokenize_df[x]))


# ## Slit data into train,validation and test

# In[ ]:


x_train = pd.Series(clean_train)
y_train = pd.Series(df_train['sentiment'])


# In[ ]:


x_val = pd.Series(clean_dev)
y_val = pd.Series(df_dev['sentiment'])


# In[ ]:


x_test = pd.Series(clean_test)
y_test = pd.Series(df_test['sentiment'])


# ## Training build_vocav with 2 method of skipgam and CBOW

# In[ ]:


def labelize_text_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


# In[ ]:


all_x = pd.concat([x_train])
all_x_w2v = labelize_text_ug(all_x, 'all')


# In[ ]:


all_x_w2v


# In[ ]:


cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)


# In[ ]:


model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)\n    model_ug_cbow.alpha -= 0.002\n    model_ug_cbow.min_alpha = model_ug_cbow.alpha')


# In[ ]:


model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)\n    model_ug_sg.alpha -= 0.002\n    model_ug_sg.min_alpha = model_ug_sg.alpha')


# #### Save results

# In[ ]:


model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')


# ## Get Keyed Vectors

# 

# In[ ]:


model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')


# ### Appending cbow and sg for better result

# In[ ]:


embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


np.append(model_ug_cbow.wv['slide'],model_ug_sg.wv['slide'])


# Now, we got our dictionary for taking a task

# ## Vectorize words into numberic (float)

# ### Build reference for our data

# In[ ]:


tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ')
tokenizer.fit_on_texts(x_train)


# - Apply it on our data
# - Check result

# In[ ]:


sequences_digit = tokenizer.texts_to_sequences(x_train)


# Take a max length of vector (reason of all inout have to be in a same size of matrix)

# In[ ]:


length = []
for x in x_train:
    length.append(len(x.split()))
max(length)


# In[ ]:


x_train_seq = pad_sequences(sequences_digit, maxlen=150)
print('Shape of data tensor:', x_train_seq.shape)


# In[ ]:


sequences_val = tokenizer.texts_to_sequences(x_val)
x_val_seq = pad_sequences(sequences_val, maxlen=150)


# In[ ]:


print('Shape of data tensor:', x_val_seq.shape)


# In[ ]:


num_words = 10000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ##### np.array_equal(embedding_matrix[5] ,embeddings_index.get('em'))
# 
# 

# # CNN 

# ### Validating on DataSet - Dev

# In[ ]:


labels_train = to_categorical(df_train.sentiment, num_classes=3)
labels_test = to_categorical(df_test.sentiment, num_classes=3)
labels_dev = to_categorical(df_dev.sentiment, num_classes=3)


# In[ ]:


model_cnn = Sequential()
e = Embedding(10000, 200, weights=[embedding_matrix], input_length=150, trainable=True)
model_cnn.add(e)
model_cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(256, activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(3, activation='sigmoid'))
model_cnn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc',f1_m,precision_m, recall_m])
history_cnn=model_cnn.fit(x_train_seq, labels_train, validation_data=(x_val_seq, labels_dev), epochs=5, batch_size=40, verbose=2)


# ### Test

# In[ ]:


sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=150)


# In[ ]:


loss_cnn, accuracy_cnn, f1_score_cnn, precision_cnn, recall_cnn = model_cnn.evaluate(x_test_seq, labels_test, verbose=0)


# In[ ]:


print(loss_cnn, accuracy_cnn, f1_score_cnn, precision_cnn, recall_cnn)


# # CNN - LSTM

# ### Validating on DataSet - Dev

# In[ ]:


model_cnn_lstm = Sequential()
e = Embedding(10000, 200, weights=[embedding_matrix], input_length=150, trainable=True)
model_cnn_lstm.add(e)
model_cnn_lstm.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_lstm.add(MaxPooling1D())
model_cnn_lstm.add(Dropout(0.2))
model_cnn_lstm.add(LSTM(300))
model_cnn_lstm.add(Dense(256, activation='relu'))
model_cnn_lstm.add(Dense(3, activation='sigmoid'))
model_cnn_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
history_cnn_lstm=model_cnn_lstm.fit(x_train_seq, labels_train, validation_data=(x_val_seq, labels_dev), epochs=5, batch_size=40, verbose=2)


# ### Test

# In[ ]:


loss_cnn_lstm, accuracy_cnn_lstm, f1_score_cnn_lstm, precision_cnn_lstm, recall_cnn_lstm = model_cnn_lstm.evaluate(x_test_seq, labels_test, verbose=0)


# In[ ]:


print(loss_cnn_lstm, accuracy_cnn_lstm, f1_score_cnn_lstm, precision_cnn_lstm, recall_cnn_lstm )


# # LSTM - CNN

# ### Validating on DataSet - Dev

# In[ ]:


from keras import backend as K

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


model_lstm_cnn = Sequential()
e = Embedding(10000, 200, weights=[embedding_matrix], input_length=150, trainable=True)
model_lstm_cnn.add(e)
model_lstm_cnn.add(LSTM(300,return_sequences=True))
model_lstm_cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_lstm_cnn.add(GlobalMaxPooling1D())
model_lstm_cnn.add(Dropout(0.2))
model_lstm_cnn.add(Dense(256, activation='relu'))
model_lstm_cnn.add(Dense(3, activation='sigmoid'))
model_lstm_cnn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc',f1_m,precision_m, recall_m])
filepath="LSTM_CNN_best_weights.{epoch:02d}-{val_acc:.41f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history_lstm_cnn=model_lstm_cnn.fit(x_train_seq, labels_train, batch_size=40, epochs=5,
                     validation_data=(x_val_seq, labels_dev),verbose=2, callbacks=[checkpoint])


# In[ ]:


model_lstm_cnn.summary()


# ### Test

# In[ ]:


loss_lstm_Cnn, accuracy_lstm_Cnn, f1_score_lstm_Cnn, precision_lstm_Cnn, recall_lstm_Cnn = model_lstm_cnn.evaluate(x_test_seq, labels_test, verbose=0)


# In[ ]:


print(loss_lstm_Cnn, accuracy_lstm_Cnn, f1_score_lstm_Cnn, precision_lstm_Cnn, recall_lstm_Cnn)


# In[ ]:


from keras.layers import Bidirectional


# In[ ]:


from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate


# In[ ]:


model_lstm_cnn_UPDATE = Sequential()
e = Embedding(10000, 200, weights=[embedding_matrix], input_length=150, trainable=True)
model_lstm_cnn_UPDATE.add(e)
model_lstm_cnn_UPDATE.add((LSTM(300,return_sequences=True,dropout=0.25, recurrent_dropout=0.1)))
model_lstm_cnn_UPDATE.add(Conv1D(filters=128, kernel_size=7, padding='same', activation='relu', strides=1))
model_lstm_cnn_UPDATE.add(MaxPooling1D())
model_lstm_cnn_UPDATE.add(Conv1D(filters=256,kernel_size=5, activation='relu',padding='same',strides=1))
model_lstm_cnn_UPDATE.add(MaxPooling1D())
model_lstm_cnn_UPDATE.add(Conv1D(filters=512, kernel_size=3, activation='relu',padding='same',strides=1))
model_lstm_cnn_UPDATE.add(MaxPooling1D())
model_lstm_cnn_UPDATE.add(Flatten())
model_lstm_cnn_UPDATE.add(Dense(256, activation='relu'))
model_lstm_cnn_UPDATE.add(Dropout(0.2))
model_lstm_cnn_UPDATE.add(Dense(3, activation='sigmoid'))
model_lstm_cnn_UPDATE.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc',f1_m,precision_m, recall_m])
filepath="LSTM_CNN_best_weights.{epoch:02d}-{val_acc:.41f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history_lstm_cnn_UPDATE=model_lstm_cnn_UPDATE.fit(x_train_seq, labels_train, batch_size=40, epochs=5,
                     validation_data=(x_val_seq, labels_dev),verbose=2, callbacks=[checkpoint])


# https://github.com/diegoschapira/CNN-Text-Classifier-using-Keras/blob/master/models.py

# In[ ]:


model_lstm_cnn_UPDATE.summary()


# In[ ]:


loss, accuracy, f1_score, precision, recall = model_lstm_cnn_UPDATE.evaluate(x_test_seq, labels_test, verbose=0)
print(loss, accuracy, f1_score, precision, recall)


# - BiLSTM-CNN (single filters) 0.1289299549046605 0.9568330157202253 0.9351887610692129 0.9361239191255311 0.9343019583823179
# - BiLSTM-CNN (multiple  filters) 0.1425636049809519 0.9547273164599739 0.9321195284023513 0.9318498289065502 0.932406822300680
# - LSTM- CNN - Sing 0.15189910863143163 0.9552537379050963 0.9259114440341426 0.9321842116414936 0.9336702464806153
# - LSTM-CNN -Multi 0.12803751011852668 0.9556748801518752 0.9335047436603925 0.933660056985637 0.9333543902097137
# - CNN 0.17975928141621947 0.9464097755736968 0.9197612686467487 0.9188679342622329 0.9207201517238244
# - CNN-LSTM 0.16549487028856935 0.944093496190997 0.9161467779232442 0.9160085565593524 0.9162981681106818

# In[ ]:


objects = ('CNN', 'LSTM-CNN', 'CNN-LSTM','LSTM-CNN Multiple Filters')
performance1 = [f1_score_cnn,f1_score_lstm_Cnn,f1_score_cnn_lstm,f1_score]
performance2 = [loss_cnn,loss_lstm_Cnn,loss_cnn_lstm,loss]
barWidth = 0.5
# Choose the height of the blue bars
y_pos = np.arange(len(objects))
r1 = np.arange(len(performance1))
plt.bar(r1, performance1, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='F1-Score')
plt.ylabel('percent')
plt.xticks(y_pos, objects)
plt.legend()
plt.ylim(0.91,0.95)
plt.title('Performance Results')
plt.show()


# In[ ]:


y_pos = np.arange(len(objects))
r1 = np.arange(len(performance1))
plt.bar(r1, performance2, width = barWidth, color =(0.625, 0.21960784494876862,    0.94117647409439087), edgecolor = 'black', capsize=7, label='Loss')
plt.ylabel('percent')
plt.xticks(y_pos, objects)
plt.legend()
plt.ylim(0.1,0.2)
plt.title('Performance Results')
plt.show()


# In[ ]:


# summarize history for accuracy
plt.plot(history_cnn.history['val_acc'], label="CNN")
#plt.plot(history_cnn.history['val_acc'])
plt.plot(history_cnn_lstm.history['val_acc'], label="CNN-LSTM")
plt.plot(history_lstm_cnn.history['val_acc'])
plt.plot(history_lstm_cnn_UPDATE.history['val_acc'])

plt.title('Validation Accurancy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['CNN', 'CNN-LSTM','LSTM-CNN','LSTM-CNN Mutiple Filters'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history_cnn.history['val_loss'])
plt.plot(history_cnn_lstm.history['val_loss'])
plt.plot(history_lstm_cnn.history['val_loss'])
plt.plot(history_lstm_cnn_UPDATE.history['val_loss'])

plt.title('Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['CNN', 'CNN-LSTM','LSTM-CNN','LSTM-CNN Mutiple Filters'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for accuracy
plt.plot(history_cnn_lstm.history['acc'])
plt.plot(history_cnn_lstm.history['val_acc'])
plt.title('CNN-LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_cnn_lstm.history['loss'])
plt.plot(history_cnn_lstm.history['val_loss'])
plt.title('CNN-LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for accuracy
plt.plot(history_lstm_cnn.history['acc'])
plt.plot(history_lstm_cnn.history['val_acc'])
plt.title('LSTM-CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_lstm_cnn.history['loss'])
plt.plot(history_lstm_cnn.history['val_loss'])
plt.title('LSTM-CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

