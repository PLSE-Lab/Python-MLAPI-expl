#!/usr/bin/env python
# coding: utf-8

# **the file cannot be run here due to time restrictions**  
# **see https://github.com/t8ch/machine-learning/blob/master/amazon-reviews/review-classification.ipynb for the complete notebook**  
# I am happy for feedback and discussion

# **here, I use the Amazon Fine Food Reviews data (https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/amazon-fine-food-reviews.zip) to**  
#     **1. build a classifier that predicts 5 star vs 1-4 star ratings**  
#     **2. build a generative model to create 5 star review text **
# 
# the work is mostly based on these resources:  
# 
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/  
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/  
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# In[ ]:


get_ipython().run_line_magic('pylab', '')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import nltk
import string
import os
import re
from __future__ import division, print_function
sns.set_style('white')


# # importing and basic preprocessing

# In[ ]:


from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[ ]:


reviews = pd.read_csv('../input/Reviews.csv')


# from https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/amazon-fine-food-reviews.zip

# In[ ]:


reviews.head()


# In[ ]:


reviews.info()


# In[ ]:


reviews.Score.unique()


# In[ ]:


reviews.Score.value_counts().plot(kind = 'bar')


# # LSTM models for review classification and generation

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nseq_length = 500 #padding/cut to this length\nnum_samples = 70000\ntexts = reviews.iloc[:num_samples].Text\n#remove html line breaks\ntext = array([re.sub(\'<[^<]+?>\', \'\', x) for x in texts])\n\n#labels = reviews.iloc[:50000].Score-1 \nlabels = reviews.Score.apply(lambda x: x>4).iloc[:num_samples] #shift to start counting with 0\n\ntokenizer = Tokenizer(num_words = 10000)\ntokenizer.fit_on_texts(texts)\nsequences = tokenizer.texts_to_sequences(texts)\n\nword_index = tokenizer.word_index\nprint(\'Found %s unique tokens.\' % len(word_index))\n\ndata = pad_sequences(sequences, maxlen= seq_length)\n\nlabels = labels.astype(int)\n#labels = to_categorical(np.asarray(labels), num_classes= 5)\nprint(\'Shape of data tensor:\', data.shape)\nprint(\'Shape of label tensor:\', labels.shape)\n\n# split the data into a training set and a validation set\nindices = np.arange(data.shape[0])\nnp.random.shuffle(indices)\ndata = data[indices]\nlabels = labels[indices]\nnb_validation_samples = int(.1 * data.shape[0])\n\nx_train = data[:-nb_validation_samples]\ny_train = labels[:-nb_validation_samples]\nx_val = data[-nb_validation_samples:]\ny_val = labels[-nb_validation_samples:]\n\nprint("fraction 5 star in sample: ", sum(labels)/num_samples)\nprint("fraction 5 star in test set: ", mean(y_val))')


# In[ ]:


data[1]


# ### GloVe embedding

# GloVe from https://nlp.stanford.edu/projects/glove/

# In[ ]:


embeddings_index = {}
f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# embedding matrix

# In[ ]:


EMBEDDING_DIM = 100 #given GloVe fileb
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length = seq_length,
                            trainable=True,
                            )


# ### build the model

# In[ ]:


# create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(embedding_layer)
model.add(Embedding(10000, embedding_vecor_length, input_length = data.shape[1]))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=64)


# ## ideas for improvement
# 
# - different embedding (GloVe didn't seem to help)
# - more layers
# - different architecture (Conv1D did not give better results)
# - hyperparameters
# - ...?

# # generate review text

# generate data (only 5 star reviews)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntext = reviews[reviews.Score == 5].iloc[:5000].Text\n# periods as words\n#text = array([string.replace(x, \'.\', " .") for x in text])\n\n#remove html line breaks\ntext = array([re.sub(\'<[^<]+?>\', \'\', x) for x in text])\n\nnum_words = 800\ntokenizer = Tokenizer(num_words = num_words, filters=\'!"#$%&()*+,.-/:;<=>?@[\\\\]^_`{|}~\\t\\n)\')\ntokenizer.fit_on_texts(text)\nconcat_text = tokenizer.texts_to_sequences(text)\nconcat_text = array([item for sublist in concat_text for item in sublist])\n\nword_index = tokenizer.word_index\nprint(\'Found %s unique tokens.\' % len(word_index))\n\n# for later text generation: int->word dictionary\ninv_word_index = {v: k for k, v in word_index.iteritems()}\n\n## training sequences of length 5\nseq_length = 6\ndataX = []\ndataY = []\nfor i in range(0, len(concat_text) - seq_length, 1):\n    seq_in = concat_text[i:i + seq_length]\n    seq_out = concat_text[i + seq_length]\n    dataX.append(seq_in)\n    dataY.append(seq_out)\nn_patterns = len(dataX)\nprint("Total Patterns: ", n_patterns)')


# ### GloVe embedding

# In[ ]:


embeddings_index = {}
f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# embedding matrix

# In[ ]:


EMBEDDING_DIM = 100 #given GloVe fileb
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length = seq_length,
                            trainable=True,
                            )


# ### build model

# In[ ]:


# Small LSTM Network to Generate Text for Alice in Wonderland
# reshape X to be [samples, time steps, features]
#X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = numpy.reshape(dataX, (n_patterns, seq_length)) #if embedding is first layer
# normalize
#X = X / float(num_words)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
embedding_vecor_length = 16
model = Sequential()
model.add(embedding_layer)
# add second LSTM layer
#if True:
    #model.add(Embedding(num_words, embedding_vecor_length, input_length = X.shape[1]))
    #model.add(LSTM(128, return_sequences = True))#, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dropout(0.2))
model.add(LSTM(256))#, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(GaussianNoise(.15))
model.add(Dense(y.shape[1], activation='softmax'))


# fit

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs= 4, batch_size = 128, callbacks=callbacks_list)


# ### text generation

# In[ ]:


# load the network weights
filename = "weights-improvement-03-4.2355.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


# pick a random seed
for _ in range(5):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("seed:" , ' '.join([inv_word_index[value] for value in pattern]), "...")
    # generate characters
    for i in range(50):
        x = numpy.reshape(pattern, (1, len(pattern)))
        #prediction = model.predict(x, verbose=0)
        #index = numpy.argmax(prediction)
        p = model.predict_proba(x, verbose = 0)[0]
        p[1] /= 1
        top_ind = argsort(p)[::-1][:5] #extract lergest probabilities
        #print top_ind
        p = p[top_ind]
        p /= sum(p)
        #print p
        index = random.choice(top_ind, 1, p = p)[0]
        #print sum(prediction[0]**2)
        result = inv_word_index[index]
        seq_in = [inv_word_index[value] for value in pattern]
        sys.stdout.write(result+' ')
        pattern = append(pattern, index)
        pattern = pattern[1:len(pattern)]
        #print pattern
    print("\n")


# ## ideas for improvement
# 
# - split into single sentences and train on them
# - different network arcitecture?
# - ...?
