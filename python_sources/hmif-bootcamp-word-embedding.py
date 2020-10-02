#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install PySastrawi


# In[ ]:


get_ipython().system("ls '../input/'")


# # Introduction to Text

# In[ ]:


import pandas as pd
import numpy as np


# ## Import Data

# In[ ]:


data = pd.read_csv('../input/review-lapak-sentiment/train.csv')


# In[ ]:


data.head(10)


# In[ ]:


data['label'].value_counts()


# ## Preprocess

# In[ ]:


# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
 
# factory = StopWordRemoverFactory()
# stopword = factory.create_stop_word_remover()

# def stop_words_removal(text):
#     return stopword.remove(text)

# print(stop_words_removal("Dengan Menggunakan Python dan Library Sastrawi saya dapat melakukan proses Stopword Removal"))


# In[ ]:


# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# def stemming(text):
#     return stemmer.stem(text)

# print(stemming("Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan"))


# In[ ]:


import itertools
def remove_repeating_character(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

remove_repeating_character("Halooo, duniaa!!")


# In[ ]:


import re
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

def lowercase(text):
    return text.lower()

print(remove_nonaplhanumeric("Halooo,,,,, duniaa!!"))
print(lowercase("Halooo, duniaa!"))


# In[ ]:


def preprocess(text):
    text = lowercase(text)
    text = remove_nonaplhanumeric(text)
#     text = stop_words_removal(text)
#     text = stemming(text)
    text = remove_repeating_character(text)
    return text


# In[ ]:


data['review_sangat_singkat'] = data['review_sangat_singkat'].apply(lambda x: preprocess(x))
data.head()


# ## Extra Preprocessing

# In[ ]:


unique_string = set()
for x in data['review_sangat_singkat']:
    for y in x.split():
        unique_string.add(y)


# In[ ]:


len(unique_string)


# In[ ]:


len_data = [len(x.split()) for x in data['review_sangat_singkat']]
print(np.mean(len_data))
print(np.median(len_data))
print(np.std(len_data))
print(np.min(len_data))
print(np.max(len_data))
print(np.percentile(len_data, 98))


# In[ ]:


embed_size = 100 # how big is each word vector
max_features = 23000 # how many unique words to use
maxlen = 20 # max number of words in a comment to use


# Tokenizer: https://keras.io/preprocessing/text/

# In[ ]:


# Example
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(["ini sebuah kalimat hehe"])
example = tokenizer.texts_to_sequences(["ini contoh kalimat juga"])
print(example[0])


# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['review_sangat_singkat'])
list_tokenized_train = tokenizer.texts_to_sequences(data['review_sangat_singkat'].values)


# In[ ]:


list_tokenized_train[0]


# Pad sequences: https://keras.io/preprocessing/sequence/

# In[ ]:


# Example
from keras.preprocessing.sequence import pad_sequences
pad_sequences(example, maxlen=maxlen)


# In[ ]:


X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)


# In[ ]:


X_t[0]


# ## Feature Engineering

# Word Embedding: https://www.kaggle.com/ilhamfp31/word2vec-100-indonesian

# In[ ]:


import gensim
DIR_DATA_MISC = "../input/word2vec-100-indonesian"
path = '{}/idwiki_word2vec_100.model'.format(DIR_DATA_MISC)
id_w2v = gensim.models.word2vec.Word2Vec.load(path)
print(id_w2v.most_similar('itb'))


# In[ ]:


index2word_set = set(id_w2v.wv.index2word)


# In[ ]:


word_index = tokenizer.word_index
nb_words = max_features
embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
for word, i in word_index.items():
    cur = word
    if cur in index2word_set:
        embedding_matrix[i] = id_w2v[cur]
        continue
        
    embedding_matrix[i] = unknown_vector


# ## Model

# In[ ]:


from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import callbacks

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(32, return_sequences=True))(x)
    x2 = Bidirectional(GRU(32, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    x = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model


# In[ ]:


from sklearn.model_selection import KFold
def get_kfold():
    return KFold(n_splits=5, shuffle=True, random_state=1)


# In[ ]:


X = X_t
y = data["label"].values

pred_cv = np.zeros(len(y))
count = 0

for train_index, test_index in get_kfold().split(X, y):
    count += 1
    print(count, end='')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    es = callbacks.EarlyStopping(monitor='val_f1', min_delta=0.0001, patience=8,
                                             verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)
    
    model = get_model()
    model.fit(X_train, 
             y_train, batch_size=16, epochs=4,
             validation_data=(X_test, y_test),
             callbacks=[es, rlr],
             verbose=1)
    
    pred_cv[[test_index]] += model.predict(X_test)[:,0]


# In[ ]:




