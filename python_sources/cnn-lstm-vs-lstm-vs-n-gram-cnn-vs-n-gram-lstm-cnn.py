#!/usr/bin/env python
# coding: utf-8

# # Analysis on pre trained embedding which will give best baseline results 
# # Full Depth analysis 

# ![](https://www.pyimagesearch.com/wp-content/uploads/2017/12/not_santa_detector_dl_logos.jpg)
# **This notebook attempts to tackle this classification problem by using Keras LSTM. While there are many notebook out there that are already tackling using this approach, I feel that there isn't enough explanation to what is going on each step. As someone who has been using vanilla Tensorflow, and recently embraced the wonderful world of Keras, I hope to share with fellow beginners the intuition that I gained from my research and study. **
# 
# **Join me as we walk through it. **

# **Notebook Objective:**
# 
# Objective of the notebook is to look at the different pretrained embeddings provided in the dataset and to see how they are useful in the model building process. 
# 
# First let us import the necessary modules and read the input data.

# In this kernel, we shall see if pretrained embeddings like Word2Vec, GLOVE and Fasttext, which are pretrained using billions of words could improve our accuracy score as compared to training our own embedding. We will compare the performance of models using these pretrained embeddings against the baseline model that doesn't use any pretrained embeddings in my previous kernel [here](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras).
# 
# ![](https://qph.fs.quoracdn.net/main-qimg-3e812fd164a08f5e4f195000fecf988f)
# 
# Perhaps it's a good idea to briefly step in the world of word embeddings and see what's the difference between Word2Vec, GLOVE and Fasttext.
# 
# Embeddings generally represent geometrical encodings of words based on how frequently appear together in a text corpus. Various implementations of word embeddings described below differs in the way as how they are constructed.
# 
# **Word2Vec**
# 
# The main idea behind it is that you train a model on the context on each word, so similar words will have similar numerical representations.
# 
# Just like a normal feed-forward densely connected neural network(NN) where you have a set of independent variables and a target dependent variable that you are trying to predict, you first break your sentence into words(tokenize) and create a number of pairs of words, depending on the window size. So one of the combination could be a pair of words such as ('cat','purr'), where cat is the independent variable(X) and 'purr' is the target dependent variable(Y) we are aiming to predict.
# 
# We feed the 'cat' into the NN through an embedding layer initialized with random weights, and pass it through the softmax layer with ultimate aim of predicting 'purr'. The optimization method such as SGD minimize the loss function "(target word | context words)" which seeks to minimize the loss of predicting the target words given the context words. If we do this with enough epochs, the weights in the embedding layer would eventually represent the vocabulary of word vectors, which is the "coordinates" of the words in this geometric vector space.
# 
# ![](https://i.imgur.com/R8VLFs2.png)
# 
# The above example assumes the skip-gram model. For the Continuous bag of words(CBOW), we would basically be predicting a word given the context. 
# 
# **GLOVE**
# 
# GLOVE works similarly as Word2Vec. While you can see above that Word2Vec is a "predictive" model that predicts context given word, GLOVE learns by constructing a co-occurrence matrix (words X context) that basically count how frequently a word appears in a context. Since it's going to be a gigantic matrix, we factorize this matrix to achieve a lower-dimension representation. There's a lot of details that goes in GLOVE but that's the rough idea.
# 
# 

# In[ ]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# Next steps are as follows:
#  * Split the training dataset into train and val sample. Cross validation is a time consuming process and so let us do simple train val split.
#  * Fill up the missing values in the text column with '_na_'
#  * Tokenize the text column and convert them to vector sequences
#  * Pad the sequence as needed - if the number of words in the text is greater than 'max_len' trunacate them to 'max_len' or if the number of words in the text is lesser than 'max_len' add zeros for remaining values.

# In[ ]:


## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=104)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 40000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


# # plotting the length of question we have

# In[ ]:


totalNumWords = [len(one_comment) for one_comment in train_X]


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.xlabel("Distribution of comment")
plt.ylabel("no of comments")
plt.title("no of comments vs no of words distribution ")
plt.show()


# # Embedding Analysis Started down

# We have four different types of embeddings.
#  * GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
#  * glove.840B.300d - https://nlp.stanford.edu/projects/glove/
#  * paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
#  * wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
#  
#  A very good explanation for different types of embeddings are given in this [kernel](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge). Please refer the same for more details..
# 
# **Glove Embeddings:**
# 
# In this section, let us use the Glove embeddings and rebuild the GRU model.

# Since we are going to evaluate a few word embeddings, let's define a function so that we can run our experiment properly. I'm going to put some comments in this function below for better intuitions.
# 
# Note that there are quite a few GLOVE embeddings in Kaggle datasets, and I feel that it would be more applicable to use the one that was trained based on Twitter text. Since the comments in our dataset consists of casual, user-generated short message, the semantics used might be very similar. Hence, we might be able to capture the essence and use it to produce a good accurate score.
# 
# Similarly, I have used the Word2Vec embeddings which has been trained using Google Negative News text corpus, hoping that it's negative words can work better in our "toxic" context.

# The function would return a new embedding matrix that has the loaded weights from the pretrained embeddings for the common words we have, and randomly initialized numbers that has the same mean and standard deviation for the rest of the weights in this matrix.

# With the embedding weights, we can proceed to build a LSTM layer. The whole architecture is pretty much the same as the previous one I have done in the earlier kernel here, except that I have turned the LSTM into a bidirectional one, and added a dropout factor to it. 
# 
# We start off with defining our input layer. By indicating an empty space after comma, we are telling Keras to infer the number automatically.

# Next, we pass it to a LSTM unit. But this time round, we will be using a Bidirectional LSTM instead because there are several kernels which shows a decent gain in accuracy by using Bidirectional LSTM.
# 
# How does Bidirectional LSTM work? 
# 
# ![](https://i.imgur.com/jaKiP0S.png)
# 
# Imagine that the LSTM is split between 2 hidden states for each time step. As the sequence of words is being feed into the LSTM in a forward fashion, there's another reverse sequence that is feeding to the different hidden state at the same time. You might noticed later at the model summary that the output dimension of LSTM layer has doubled to 120 because 60 dimensions are used for forward, and another 60 are used for reverse.
# 
# The greatest advantage in using Bidirectional LSTM is that when it runs backwards you preserve information from the future and using the two hidden states combined, you are able in any point in time to preserve information from both past and future.
# 

# We are also introducing 2 more new mechanisms in this notebook: **LSTM Drop out and recurrent drop out.**
# 
# Why are we using dropout? You might have noticed that it's easy for LSTM to overfit, and in my previous notebook, overfitting problem starts to surface in just 2 epochs! Drop out is not something new to most of us, and these mechanisms applies the same dropout principles in a LSTM context.
# 
# ![](https://i.imgur.com/ksSyArD.png)
# LSTM Dropout is a probabilistic drop out layer on the inputs in each time step, as depict on the left diagram(arrows pointing upwards). On the other hand, recurrent drop out is something like a dropout mask that applies drop out between the hidden states throughout the recursion of the whole LSTM network, which is depicted on the right diagram(arrows pointing to the right). 
# 
# These mechanisms could be set via the "dropout" and "recurrent_dropout" parameters respectively. Please ignore the colors in the picture.

# # Architecture is same only dimensions are changed 

# This is the architecture of the model we are trying to build. It's always to good idea to list out the dimensions of each layer in the model to think visually and help you to debug later on.
# ![](https://i.imgur.com/txJomEa.png)

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.3)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(train_X, train_y, batch_size=1024, epochs=2, validation_data=(val_X, val_y))


# In[ ]:


pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
max_glove=[]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    max_glove.append(metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
max_glove=max(max_glove)


# Results seem to be better than the model without pretrained embeddings.

# In[ ]:


pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)


# In[ ]:


del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)


# **Wiki News FastText Embeddings:**
# 
# Now let us use the FastText embeddings trained on Wiki News corpus in place of Glove embeddings and rebuild the model.

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.3)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(train_X, train_y, batch_size=1024, epochs=2, validation_data=(val_X, val_y))


# In[ ]:


pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)
max_fast_text=[]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    max_fast_text.append(metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))


# In[ ]:


pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)


# In[ ]:


del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)


# **Paragram Embeddings:**
# 
# In this section, we can use the paragram embeddings and build the model and make predictions.

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.3)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))


# In[ ]:


pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)
max_paragram=[]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    max_paragram.append(metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))))


# In[ ]:


pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)


# In[ ]:


del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)


# **Observations:**
#  * Overall pretrained embeddings seem to give better results comapred to non-pretrained model. 
#  * The performance of the different pretrained embeddings are almost similar.
#  
# **Final Blend:**
# 
# Though the results of the models with different pre-trained embeddings are similar, there is a good chance that they might capture different type of information from the data. So let us do a blend of these three models by averaging their predictions.

# In[ ]:


pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y
max_blend=[]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    max_blend.append(metrics.f1_score(val_y, (pred_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))


# The result seems to better than individual pre-trained models and so we let us create a submission file using this model blend.

# In[ ]:


max_paragram


# In[ ]:


max_paragram=max(max_paragram)
max_fast_text = max(max_fast_text)
max_blend = max(max_blend)
import seaborn as sns
sns.lineplot(["Glove","Fasttext","Paragram","Blend"],y=[max_glove,max_fast_text,max_paragram,max_blend])
plt.xlabel("models")
plt.ylabel("F1-Score")
plt.title("Score Board")
plt.show()


# In[ ]:


pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

