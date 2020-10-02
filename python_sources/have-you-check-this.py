#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re


# In[ ]:


def load_training_data():
    data_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
    x = data_df['Phrase'].values
    y = data_df['Sentiment'].values
    print('training data\'s len:', x.shape[0])
    return x, y


# In[ ]:


def load_testing_data():
    data_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
    x = data_df['Phrase'].values
    print('testing data\'s len:', x.shape[0])
    return x


# In[ ]:


x_train, y_train = load_training_data()


# In[ ]:


x_test = load_testing_data()


# In[ ]:


print(x_train[:5])


# In[ ]:


print(y_train[:5])


# In[ ]:


print(x_test[:5])


# In[ ]:


from keras.preprocessing.text import Tokenizer


# In[ ]:


tokenizer = Tokenizer()


# In[ ]:


tokenizer.fit_on_texts(list(x_train) + list(x_test))


# In[ ]:


x_train_seqs = tokenizer.texts_to_sequences(list(x_train))


# In[ ]:


print(x_train_seqs[:5])


# In[ ]:


word2idx = tokenizer.word_index


# In[ ]:


from keras.preprocessing.sequence import pad_sequences


# In[ ]:


x_train_paded = pad_sequences(x_train_seqs)


# In[ ]:


print(x_train_paded.shape)


# In[ ]:


print(x_train_paded[:5])


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


y_train_onehot = to_categorical(y_train)


# In[ ]:


print(y_train_onehot.shape)


# In[ ]:


print(y_train_onehot[:5])


# In[ ]:


import numpy as np


# In[ ]:


def shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]


# In[ ]:


x_train_shuffled, y_train_shuffled = shuffle(x_train_paded, 
                                             y_train_onehot)


# In[ ]:


print(x_train_shuffled[:5])


# In[ ]:


print(y_train_shuffled[:5])


# In[ ]:


from gensim.models import KeyedVectors


# In[ ]:


wv = KeyedVectors.load_word2vec_format('word2vec.6B.100d.txt')


# In[ ]:


embeddings = np.zeros((len(word2idx) + 1, 100))


# In[ ]:


'the' in wv.vocab


# In[ ]:


for word, idx in word2idx.items():
    if word in wv.vocab:
        embeddings[idx] = wv.get_vector(word)


# In[ ]:


print(embeddings[:5])


# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Activation


# In[ ]:


gru_model = Sequential()


# In[ ]:


gru_model.add(Embedding(embeddings.shape[0], 
                        100, 
                        weights=[embeddings], 
                        trainable=False))


# In[ ]:


gru_model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
gru_model.add(Dense(5, activation='softmax'))


# In[ ]:


gru_model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])


# In[ ]:


gru_model.fit(x_train_shuffled, y_train_shuffled, batch_size=256, 
              epochs=10, verbose=1)


# In[ ]:


x_test_seqs = tokenizer.texts_to_sequences(x_test)


# In[ ]:


x_test_paded = pad_sequences(x_test_seqs)


# In[ ]:


test_pred = gru_model.predict_classes(x_test_paded)


# In[ ]:


print(test_pred)


# In[ ]:


test_df = pd.read_csv('test.tsv', sep='\t')


# In[ ]:


test_df['Sentiment'] = test_pred.reshape(-1, 1)


# In[ ]:


test_df.to_csv('gru-word2vec.csv', columns=['PhraseId', 'Sentiment'], 
               index=False, header=True)


# In[ ]:




