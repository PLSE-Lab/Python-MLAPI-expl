#!/usr/bin/env python
# coding: utf-8

# ## __Text Mining on Reviews__
# 
# ![](http://)![](http://thepayoffprinciple.com/wp-content/uploads/2018/06/NLP-Practitioner.jpg)

# [](http://)This notebook is __a lot inspired__ by : 
# - https://github.com/m2dsupsdlclass/lectures-labs
# - [](http://)https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/NLP_word_vectors_classification_rendered.ipynb
# 
# It is part of an amazing github created by Olivier Grisel and Charles Ollion for their courses at Master Data Science from Polytechnique
# 
# The goal of this notebook is to learn to use Neural Networks for text classification. The main goal is for you to understand how we can apply deep learning on raw text and what are the techniques behin it
# 
# In this notebook, we will:
# - Train a shallow model with learning embeddings
# 
# However keep in mind:
# - Deep Learning can be better on text classification that simpler ML techniques, but only on very large datasets and well designed/tuned models.
# - Many open source projects are really powerfull annd can be re-used: [word2vec](https://github.com/dav/word2vec) and [gensim's word2vec](https://radimrehurek.com/gensim/models/word2vec.html)   (self-supervised learning only), [fastText](https://github.com/facebookresearch/fastText) (both supervised and self-supervised learning), [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki) (supervised learning).
# - Plain shallow sparse TF-IDF bigrams features without any embedding and Logistic Regression or Multinomial Naive Bayes is often competitive in small to medium datasets.

# In[ ]:


# libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
np.random.seed(32)


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical


get_ipython().run_line_magic('matplotlib', 'inline')


# ## The dataset

# In[ ]:


df = pd.read_csv("../input/GrammarandProductReviews.csv")


# In[ ]:


df.head()


# We will only consider the  text of the reviews and the ratings.
# 
# We are going to make an approximation in order to predict from the text the satisfaction level of the customer.

# In[ ]:


plt.hist(df['reviews.rating'])


# Due to the distribution of the ratings, we will consider that a customer is pleased by the product if the rating is higher than 3. Thus we will consider that a customer doesn't make a good review when the rating is equal or lower to 3.

# In[ ]:


df['target'] = df['reviews.rating']<4


# In[ ]:


plt.hist(df['target'])


# We can see that we have a lot of "happy" customer due to our target distribution

# In[ ]:


train_text, test_text, train_y, test_y = train_test_split(df['reviews.text'],df['target'],test_size = 0.2)


# In[ ]:


train_text.shape


# ### Preprocessing text for the (supervised) CBOW model
# 
# We will implement a simple classification model in Keras. Raw text requires (sometimes a lot of) preprocessing.
# 
# The following cells uses Keras to preprocess text:
# - using a tokenizer. You may use different tokenizers (from scikit-learn, NLTK, custom Python function etc.). This converts the texts into sequences of indices representing the `20000` most frequent words
# - sequences have different lengths, so we pad them (add 0s at the end until the sequence is of length `1000`)
# - we convert the output classes as 1-hot encodings

# In[ ]:


MAX_NB_WORDS = 20000

# get the raw text data
texts_train = train_text.astype(str)
texts_test = test_text.astype(str)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


sequences[0]


# The tokenizer object stores a mapping (vocabulary) from word strings to token ids that can be inverted to reconstruct the original message (without formatting):

# In[ ]:


type(tokenizer.word_index), len(tokenizer.word_index)


# In[ ]:


index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())


# In[ ]:


" ".join([index_to_word[i] for i in sequences[0]])


# 
# Let's have a closer look at the tokenized sequences:

# In[ ]:


seq_lens = [len(s) for s in sequences]
print("average length: %0.1f" % np.mean(seq_lens))
print("max length: %d" % max(seq_lens))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.hist(seq_lens, bins=50);


# 
# Let's zoom on the distribution of regular sized posts. The vast majority of the posts have less than 200 symbols:

# In[ ]:


plt.hist([l for l in seq_lens if l < 200], bins=50);


# In[ ]:


MAX_SEQUENCE_LENGTH = 150

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)


# In[ ]:


y_train = train_y
y_test = test_y

y_train = to_categorical(np.asarray(y_train))
print('Shape of label tensor:', y_train.shape)


# ## A simple supervised CBOW model in Keras

# Vector space model is well known in information retrieval where each document is represented as a vector. The vector components represent weights or importance of each word in the document. The similarity between two documents is computed using the cosine similarity measure.
# 
# ![](https://iksinc.files.wordpress.com/2015/04/screen-shot-2015-04-12-at-10-58-21-pm.png?w=768&h=740)
# 
# image & explanation taken from : https://iksinc.online/tag/continuous-bag-of-words-cbow/

# In[ ]:


from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

EMBEDDING_DIM = 50
N_CLASSES = 2

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='softmax')(average)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])


# In[ ]:


model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=10, batch_size=128)


# In[ ]:


output_test = model.predict(x_test)
print("test auc:", roc_auc_score(y_test,output_test[:,1]))


# ## A complex model : LSTM

# ![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
# 
# image taken from : http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# In[ ]:


# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
predictions = Dense(2, activation='softmax')(x)


model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[ ]:


model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=2, batch_size=128)


# In[ ]:


output_test = model.predict(x_test)
print("test auc:", roc_auc_score(y_test,output_test[:,1]))


# ## A more complex model : CNN - LSTM

# In[ ]:


# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# 1D convolution with 64 output channels
x = Conv1D(64, 5)(embedded_sequences)
# MaxPool divides the length of the sequence by 5
x = MaxPooling1D(5)(x)
x = Dropout(0.2)(x)
x = Conv1D(64, 5)(x)
x = MaxPooling1D(5)(x)
# LSTM layer with a hidden size of 64
x = Dropout(0.2)(x)
x = LSTM(64)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[ ]:


model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=5, batch_size=128)


# In[ ]:


output_test = model.predict(x_test)
print("test auc:", roc_auc_score(y_test,output_test[:,1]))


# The model seems to overfit. Indeed the train error is really low whereas the test error is really higher. It seems that we have a variance porblem. Adding regularization like drop-out may help to stabilize the performance
# 
# Edit : we have added drop-out. It is still not enough. We need to work on regularization technics to stabilize our performance.
# 
# Edit2 : With less epochs, we manage to reduce the variance.

# ### Visualize the outputs of our own Embeddings

# We are going to use our precedent model for our embedding. Then we will pass our 100 first reviews in the embedding and plot them with the label.

# In[ ]:


from keras import backend as K
get_emb_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].input])
embedding_output = get_emb_layer_output([x_test[:3000]])[0]


# In[ ]:


emb_shape = embedding_output.shape
to_plot_embedding = embedding_output.reshape(emb_shape[0],emb_shape[1]*emb_shape[2])
y = y_test[:3000]


# to visualize our results we will use tsne

# In[ ]:


sentence_emb_tsne = TSNE(perplexity=30).fit_transform(to_plot_embedding)


# In[ ]:


print(sentence_emb_tsne.shape)
print(y.shape)


# In[ ]:


plt.figure()
plt.scatter(sentence_emb_tsne[np.where(y == 0), 0],
                   sentence_emb_tsne[np.where(y == 0), 1],
                   marker='x', color='g',
                   linewidth='1', alpha=0.8, label='Happy')
plt.scatter(sentence_emb_tsne[np.where(y == 1), 0],
                   sentence_emb_tsne[np.where(y == 1), 1],
                   marker='v', color='r',
                   linewidth='1', alpha=0.8, label='Unhappy')

plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show()  


# We can definitly see a trand in our representation between negative and postive sentences.

# In[ ]:




