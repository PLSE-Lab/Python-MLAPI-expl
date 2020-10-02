#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train = train.fillna('empty')
test = test.fillna('empty')


# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())


# In[ ]:


for i in range(6):
    print(train.question1[i])
    print(train.question2[i])
    print()


# In[ ]:


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\'+-=]", " ", text)
    text = re.sub(r"\'s", " 's ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


# In[ ]:


def process_questions(question_list, questions, question_list_name, dataframe):
# function to transform questions and display progress
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


# In[ ]:


train_question1 = []
process_questions(train_question1, train.question1, 'train_question1', train)


# In[ ]:


train_question1[:10]


# In[ ]:


train_question2 = []
process_questions(train_question2, train.question2, 'train_question2', train)


# In[ ]:


test_question1 = []
process_questions(test_question1, test.question1, 'test_question1', test)


# In[ ]:




test_question2 = []
process_questions(test_question2, test.question2, 'test_question2', test)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import datetime, time, json
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed,                          Lambda, Activation, LSTM, Flatten, Bidirectional, Convolution1D, GRU, MaxPooling1D,                          Convolution2D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from collections import defaultdict


# In[ ]:


word_count = defaultdict(int)


# In[ ]:


for question in train_question1:
    word_count[question] += 1
print("train_question1 is complete.")
    
for question in train_question2:
    word_count[question] += 1
print("train_question2 is complete")

for question in test_question1:
    word_count[question] += 1
print("test_question1 is complete.")

for question in test_question2:
    word_count[question] += 1
print("test_question2 is complete")

print("Total number of unique words:", len(word_count))


# In[ ]:


lengths = []
for question in train_question1:
    lengths.append(len(question.split()))

for question in train_question2:
    lengths.append(len(question.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# In[ ]:


lengths.counts.describe()


# In[ ]:


np.percentile(lengths.counts, 99.5)


# In[ ]:


num_words = 200000

train_questions = train_question1 + train_question2
tokenizer = Tokenizer(nb_words = num_words)
tokenizer.fit_on_texts(train_questions)
print("Fitting is compelte.")
train_question1_word_sequences = tokenizer.texts_to_sequences(train_question1)
print("train_question1 is complete.")
train_question2_word_sequences = tokenizer.texts_to_sequences(train_question2)
print("train_question2 is complete")


# In[ ]:


test_question1_word_sequences = tokenizer.texts_to_sequences(test_question1)
print("test_question1 is complete.")
test_question2_word_sequences = tokenizer.texts_to_sequences(test_question2)
print("test_question2 is complete.")


# In[ ]:


word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))


# In[ ]:


max_question_len = 37

train_q1 = pad_sequences(train_question1_word_sequences, 
                              maxlen = max_question_len,
                              padding = 'post',
                              truncating = 'post')
print("train_q1 is complete.")

train_q2 = pad_sequences(train_question2_word_sequences, 
                              maxlen = max_question_len,
                              padding = 'post',
                              truncating = 'post')
print("train_q2 is complete.")


# In[ ]:


test_q1 = pad_sequences(test_question1_word_sequences, 
                             maxlen = max_question_len,
                             padding = 'post',
                             truncating = 'post')
print("test_q1 is complete.")

test_q2 = pad_sequences(test_question2_word_sequences, 
                             maxlen = max_question_len,
                             padding = 'post',
                             truncating = 'post')
print("test_q2 is complete.")


# In[ ]:


y_train = train.is_duplicate


# In[ ]:


embeddings_index = {}
with open('glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))


# In[ ]:





# Load GloVe to use pretrained vectors
# From this link: https://nlp.stanford.edu/projects/glove/


# In[50]:

# Need to use 300 for embedding dimensions to match GloVe vectors.
embedding_dim = 300

nb_words = len(word_index)
word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))


# In[66]:

units = 150
dropout = 0.25
nb_filter = 32
filter_length = 3
embedding_dim = 300

model1 = Sequential()
model1.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model1.add(Convolution1D(nb_filter = nb_filter, 
                        filter_length = filter_length, 
                        border_mode = 'same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Convolution1D(nb_filter = nb_filter, 
                        filter_length = filter_length, 
                        border_mode = 'same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Flatten())



model2 = Sequential()
model2.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))

model2.add(Convolution1D(nb_filter = nb_filter, 
                        filter_length = filter_length, 
                        border_mode = 'same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))

model2.add(Convolution1D(nb_filter = nb_filter, 
                        filter_length = filter_length, 
                        border_mode = 'same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))

model2.add(Flatten())



model3 = Sequential()
model3.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))
model3.add(TimeDistributed(Dense(embedding_dim)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(dropout))
model3.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))


model4 = Sequential()
model4.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_question_len,
                     trainable = False))
model4.add(TimeDistributed(Dense(embedding_dim)))
model4.add(BatchNormalization())
model4.add(Activation('relu'))
model4.add(Dropout(dropout))
model4.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))


modela = Sequential()
modela.add(Merge([model1, model2], mode='concat'))
modela.add(Dense(units))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))

modela.add(Dense(units))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))


modelb = Sequential()
modelb.add(Merge([model3, model4], mode='concat'))
modelb.add(Dense(units))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))

modelb.add(Dense(units))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))


model = Sequential()
model.add(Merge([modela, modelb], mode='concat'))
model.add(Dense(units))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
#sgd = SGD(lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[67]:

save_best_weights = 'question_pairs_weights.h5'

t0 = time.time()
callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
history = model.fit([train_q1, train_q2],
                    y_train,
                    batch_size=200,
                    nb_epoch=100,
                    validation_split=0.1,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))



# In[68]:

summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                              'train_acc': history.history['acc'],
                              'valid_acc': history.history['val_acc'],
                              'train_loss': history.history['loss'],
                              'valid_loss': history.history['val_loss']})


# In[69]:

summary_stats


# In[70]:

plt.plot(summary_stats.train_loss)
plt.plot(summary_stats.valid_loss)
plt.show()


# In[71]:

min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
print('Minimum loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))
min_loss = round(min_loss, 4)


# In[72]:

model.load_weights(save_best_weights)
predictions = model.predict([test_q1, test_q2], verbose = True)


# In[73]:

#Create submission
submission = pd.DataFrame(predictions, columns=['is_duplicate'])
submission.insert(0, 'test_id', test.test_id)
file_name = 'submission_{}.csv'.format(min_loss)
submission.to_csv(file_name, index=False)


# In[74]:

submission.head(10)

