#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import bz2
import gc
import chardet
import re
import os
import pickle
print(os.listdir("../input"))


# In[ ]:


from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, Dropout, concatenate, Layer, InputSpec, CuDNNLSTM
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# # Read & Preprocess data

# In[ ]:


train_file = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2')
test_file = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2')


# ## Create Lists containing Train & Test sentences

# In[ ]:


train_file_lines = train_file.readlines()
test_file_lines = test_file.readlines()


# In[ ]:


del train_file, test_file


# ## Convert from raw binary strings to strings that can be parsed

# In[ ]:


train_file_lines = [x.decode('utf-8') for x in train_file_lines]
test_file_lines = [x.decode('utf-8') for x in test_file_lines]


# In[ ]:


train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
    
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])
                                                       
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
        
for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])


# In[ ]:


del train_file_lines, test_file_lines
train_sentences[0]


# In[ ]:


gc.collect()


# In[ ]:


# Create train and test dataframes
#Na_train = {'Sentence': train_sentences, 'Label': train_labels}
#Nav_train = pd.DataFrame(Na_train)

#Na_test = {'Sentence': test_sentences, 'Label': test_labels}
#Nav_test = pd.DataFrame(Na_test)

#Nav_train.head()
#Nav_train.to_csv("amazon_train.csv")


# ### Separate Positive and Negative tweets

# In[ ]:


#train_pos = Nav_train[Nav_train['Label'] == 1]
#train_pos = Nav_train['Sentence']
#train_neg = Nav_train[Nav_train['Label'] == 0]
#train_neg = Nav_train['Sentence']

#test_pos = Nav_test[Nav_test['Label'] == 1]
#test_pos = Nav_test['Sentence']
#test_neg = Nav_test[Nav_test['Label'] == 0]
#test_neg = Nav_test['Sentence']


# 
# ## Cleaning and Feature Extraction

# # 1: Using Logistic Regression
# 
# ## Vectorization of data

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary = True)
cv.fit(train_sentences)
log_train=cv.transform(train_sentences)
log_test =cv.transform(test_sentences)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


#for c in [0.01, 0.05, 0.025, 0.5, 1]:
#lr = LogisticRegression(C=0.05)
#lr.fit(log_train, train_labels)


# In[ ]:



# save the model to disk
filename = 'finalized_model.sav'
#pickle.dump(lr, open(filename, 'wb'))
 
# some time later...


# In[ ]:



# load the model from disk
lr_model = pickle.load(open("../input/sentiment-analysis-logregre-vs-cudnnlstm/finalized_model.sav", 'rb'))


# In[ ]:



print("The product is  good: RESULT:")
testing = "the product is  good"
testing = [testing]
print(lr_model.predict(cv.transform(testing)))

print("The product is not good: RESULT:")
testing = "The product is not good"
testing = [testing]

print(lr_model.predict(cv.transform(testing)))


# In[ ]:


acc=(accuracy_score(test_labels, lr_model.predict(log_test)))
print("Accuracy using LogisticRegression is", acc)


# # 2: Using CuDNNLSTM MODEL

# In[ ]:


max_features = 20000
maxlen = 100


# In[ ]:


tokenizer = text.Tokenizer(num_words=max_features)


# In[ ]:


tokenizer.fit_on_texts(train_sentences)


# In[ ]:


tokenized_train = tokenizer.texts_to_sequences(train_sentences)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)


# In[ ]:


print(tokenized_train[0])
print(X_train[0])


# In[ ]:


tokenized_test = tokenizer.texts_to_sequences(test_sentences)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)


# In[ ]:


EMBEDDING_FILE = '../input/glovetwitter100d/glove.twitter.27B.100d.txt'


# In[ ]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))


# In[ ]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:



del tokenized_test, tokenized_train, tokenizer, train_sentences, test_sentences, word_index, embeddings_index, all_embs, nb_words
gc.collect()


# In[ ]:


batch_size = 2048
epochs = 7
embed_size = 100


# In[ ]:


gc.collect()


# In[ ]:


def cudnnlstm_model(conv_layers = 2, max_dilation_rate = 3):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, kernel_size = 3)(x)
    prefilt = Conv1D(2*embed_size, kernel_size = 3)(x)
    x = prefilt
    for strides in [1, 1, 2]:
        x = Conv1D(128*2**(strides), strides = strides, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_size=3, kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)
    x_f = CuDNNLSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)  
    x_b = CuDNNLSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)
    x = concatenate([x_f, x_b])
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model

cudnnlstm_model = cudnnlstm_model()
cudnnlstm_model.summary()


# In[ ]:


weight_path="early_weights.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
callbacks = [checkpoint, early_stopping]


# In[ ]:


#cudnnlstm_model.fit(X_train, train_labels, batch_size=batch_size, epochs=epochs, shuffle = True, validation_split=0.20, callbacks=callbacks)


# In[ ]:


cudnnlstm_model.load_weights(weight_path)
score, acc = cudnnlstm_model.evaluate(X_test, test_labels, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:



# serialize model to JSON
model_json = cudnnlstm_model.to_json()
with open("cudnnlstm_model.json", "w") as json_file:
   json_file.write(model_json)
# serialize weights to HDF5
cudnnlstm_model.save_weights("cudnnlstm_model.h5")
print("Saved model to disk")

# later...


# In[ ]:


print(os.listdir())


# In[ ]:


loaded_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])


# In[ ]:


#print(tokenized_test[0])
loaded_model.predict( sequence.pad_sequences(tokenizer.texts_to_sequences(X_test[0],maxlen=maxlen))


# # Conclusion
# The logesstic regression model gave accuracy of 90% wherease the CuDNNLSTM gave accuracy of 93%. The deep learning model using GPU to perform sentiment annalysis and can be trained with more iteration to give higher accuracy than 93%
