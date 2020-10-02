#!/usr/bin/env python
# coding: utf-8

# Here are some of my experiences with the movie review sentiment analysis dataset  ( I have also documented my other experiences with Text and NLP ).  The dataset of this competition turned out to be different (read weird here), in a sense regular Data Preparation/Cleansing and Feature harvesting didn't work. But this gave me a great learning.  I tried many possible combinations of data preparation/cleansing, feature extraction and network architectures. (Hence 108 submissions :-))
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from keras import backend
import logging

# To have reproducability: Set all the seeds, make sure multithreading is off, if possible don't use GPU. 
tf.set_random_seed(7)
np.random.seed(7)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))
pd.set_option('display.max_colwidth', -1)


train_filename =  '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
test_filename =  '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'

train_data_raw = pd.read_csv(train_filename, delimiter="\t").fillna('')
test_data_raw = pd.read_csv(test_filename, delimiter="\t").fillna('')

print('Input Files read')


# <h2> Data Understanding *(understanding some of the weirdness of this dataset) * </h2>
# 
# As you can see sentence id denotes a single review with the phrase column having the entire review text as an input instance followed by random suffixes of the same sentence to form multiple phrases with subsequent phrase ids. This repeats  for every single new sentence id (or new review per se). The sentiment is coded with 5 values 0= Very negative to 4=Very positive and everything else in between. 
# 
# **Some quirks of this dataset when compared to a typical review text.** (Unfortunately you are gonna having to trust me on the quirks :-), I am **not** cliaming expertise but I have researched  human written text reviews from sources like Amazon since 2016)
# 
# A quick glance will show you that the data is a little weird for a sentiment corpus:
# 
# <ul>
# <li>Phrases of sentences are** chopped up compeltely randomly**. So logic like sentence tokenization based on periods or punctuations or something of that sort doesn't apply</li>
# <li>Certain phrases are **one worded with a sentiment 2**. (Obviously a sentiment 2 means neutral and if we trained using the words including the stop words it would be straight forward to classify a "neutral" sentiment.)</li>
# <li>For some phrases inclusion of a punctuation like a comma or a full stop changes the sentiment from say 2 to 3 i.e neutral to positive (we just have to leave our common sense out for analysing this data set I guess :-))</li>
# <li>Some phrases **starts** with a punctuation like a **backquote**. </li>
# <li>Some phrases **end** with a **punctuation** (i.e apart from a period)</li>
# <li>There are some residues of POS tagging left, you can occassionaly see **POST TAGS like -RRB-, -LRB-** in the phrases.</li>
#  <li>Some phrases only had a **POST tag like a  -LRB- **:-)</li>
# </ul>
#     
#  I told myself that  as much as there are some weird aspects to this dataset, these can be helpful and may be predictive. Afterall, we are looking for patterns in data. Hence, i thought it would be easier for us to engineer features, I mean apart from the text features that can be extracted from the corpus. You will know shortly,  How wrong I was :-)
# 

# <h2>On Data Preparation (especially cleansing) </h2>
# 
# * Here are couple of instances where punctutaions appeared to be predictive. So if we "cleanedup" the data in the name of data preparation some predictiveness will be lost. (I know, rest-in-peace common-sense :-)) 

# In[ ]:


train_data_raw[(train_data_raw['PhraseId'] >= 517) & (train_data_raw['PhraseId'] <= 518)]


# In[ ]:


train_data_raw[(train_data_raw['PhraseId'] >= 1509) & (train_data_raw['PhraseId'] <= 1510)]


# The above observations should tell you 2 **important** things 
# 
# 1. Don't consider only words,  other tokens are important too.
# 2. Words in certain ***order*** (in conjunction with some tokens) seems predictive than just simple "bag-of-words". So we can't go with count based vectorisation methods and its cousins like TF-IDF.
# 
# But its hard to find whats more predictive just by random sampling and eyeballing the data. So I went ahead and fit a quick and dirty LogisticRegression Model with TF-IDF words and Char ngrams. My hunch was right. The training accuracy was crappy, let alone CV accuracy.
# 
# **So what should be the compositionality, some options that were infront of me **
# 
# *  (Local represenation or count based)  Use Static Word bigrams or Trigrams and retry TF-IDF ?
# *  (Distributed represenation)  Use WordEmbeddings + dynamic word n-grams that CNN1D can detect?
# * (Distributed represenation) Use WordEmbeddings +  sequence models like RNN ( LSTM or GRU )?
# * (Language Modelling) : Use different level of embeddings Phrase itself , so use something like [Phrase2Vec](http://kavita-ganesan.com/how-to-incorporate-phrases-into-word2vec-a-text-mining-approach/#.W_PDHHozZTY) or [sentence2Vec](https://github.com/stanleyfok/sentence2vec) ?
# * (Language Modelling) : Use Pretrained language model ? Like [ElMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
# * (Language Modelling): Use a pretrained sentence encoder ? [like this] (https://arxiv.org/abs/1803.11175) ?
# 
# I took the advices from[ Yoav Goldberg's book (free PDF)](https://piazza-resources.s3.amazonaws.com/iyaaxqe1yxg7fm/iybxmq5nkds6ln/goldberg_2017_book_draft_20170123.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAR6AWVCBXZCIR6MHJ%2F20181008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20181008T111048Z&X-Amz-Expires=10800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=FQoGZXIvYXdzEHsaDMjmRiJsMGoUfsmpmCK3A4trEjArtCnFNl9lsi19FBE%2FcHxdHRnTzZ2PHCiFyMYo3BEtr5LJKkWqP8IWtAyJYqvPgnG6jmgNn3LqlVBtQQhYVbEzCYtR%2B30stzaEOAi31V5h06RfjPc958mDAVMmsCMsAD3h5s4hWYnWaZanBjl33mMdReGUdJ0kP9lHlNUJB8UpNrUfo7A8DT9SqBLWloJoeGONzdEaJfOHP9Fyy96WYpiTmpPNRV6IV322w86dxW764Og3g9lAApIM2kZZFC3WKl8HXbeEs7kSd85KIiBnhne%2FJqWIi4yK1NoyznY4ScMvRKpuHZSM5xFBTsperuiSKqw6EDeU8CbrNoER9ApMdQpAeDeFl3567JUm%2Bece%2BjB%2BrD5ygV%2F6JYEgt6G2y1LsBO0DRCUWjbdPIx4NMkHJLvNzb9M%2Bllqlgi45GHodIOgajslP3%2FfYDccu%2BbFO8K0hnAn91R6OT2OGFMxnjWwRmsequUBRsAZ2KZJHaf7%2FllGIgMFzPSw9T2mzxRWFDJS0BowTGl1iKiQzNAZKYxfgxpiYg9ME8hXUZkyqsKB9Na9wncyEA60oqRD6h7Rqt91skZvk2DMovdHs3QU%3D&X-Amz-Signature=b5c9976254bc42c33e42727ef65bc0b14d4a18a4995b5c0da4df7ef76d7249b5)
# and hypothesized a dynamic n-gram detector (unlike CBOW or CBOngrams) would do well on this data set. Hence I went with a 1-D CNN's with 3 different n-gram size. In combination with RNNs (later stuck to the same window size (CNN filter) size '3'). The book explains how cool these networks are when it comes to extracting features like gappy (non-contiguous) n-grams from dynamic lengths from the corpus.
# 
# 
# 
# 

# In[ ]:


from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


vocab_size = 20000  # based on words in the entire corpus
max_len = 60        # based on word count in phrases

all_corpus   = list(train_data_raw['Phrase'].values) + list(test_data_raw['Phrase'].values)
train_phrases  = list(train_data_raw['Phrase'].values) 
test_phrases   = list(test_data_raw['Phrase'].values)
X_train_target_binary = pd.get_dummies(train_data_raw['Sentiment'])

#Vocabulary-Indexing of thetrain and test phrases, make sure "filters" parm doesn't clean out punctuations

tokenizer = Tokenizer(num_words=vocab_size, lower=True, filters='\n\t')
tokenizer.fit_on_texts(all_corpus)
encoded_train_phrases = tokenizer.texts_to_sequences(train_phrases)
encoded_test_phrases = tokenizer.texts_to_sequences(test_phrases)

#Watch for a POST padding, as opposed to the default PRE padding

X_train_words = sequence.pad_sequences(encoded_train_phrases, maxlen=max_len,  padding='post')
X_test_words = sequence.pad_sequences(encoded_test_phrases, maxlen=max_len,  padding='post')
print (X_train_words.shape)
print (X_test_words.shape)
print (X_train_target_binary.shape)

print ('Done Tokenizing and indexing phrases based on the vocabulary learned from the entire Train and Test corpus')


    


# <h2> On Text features: Some questions to reflect on </h2>
# 
# 
# <ul>
# <li>What is compositionality ? i.e what is the lowest tactical unit of our text features (text here means word or char). In other words, Should we operate at word level (or word ngram level) or should we go down to char level (char ngram level) ? </li>
# <li>Does word(char) order matter or we can get away with count based feature extraction techniques like TF-IDF ?</li>
# <li>Does word(char) context matter ? i.e. should we go for local or distributed representation of text based on word context ?</li>
# <li>If we use distributed text embeddings should we use sparse embeddings like from Counts based Word Embedding techniques: Word cooccurence matrix + PPMI + SVD or should we go for dense prediction based word embeddings like from WORD2VEC or GloVE or FASTTEXT ? (This article of Sebastian should give you a clarity on how they [perform](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/) - </li>
# <li>If we use dense word embeddings, should we use pretrained word embeddings and do a transfer learning of weights ? or should we learn our custom embeddings ?</li>
# <li>If we learn custom embeddings, should we use GENSIM Word2Vec of FastText implementation or use features like the Embedding layer of Libraries like Keras ?</li>
# </ul>
# 
# Answers to these questions partly can be obtained by probing the data, partly by running simple linear model with count based features (TF-IDF) on a sample of the data. But getting answers to these questions is imperative. It can help us in almost all the steps like data preparation/cleansing to network architecture and the choice of feature extracting networks (if we use a DNN) 
# 
# 
# 
# 
# 

# <h2> Class distributions and % Neutral Data </h2>

# In[ ]:


print ("% of neutral sentiment phrases",train_data_raw[(train_data_raw['Sentiment'] == 2)].count()[0] /train_data_raw.shape[0])


# In[ ]:


train_data_raw.groupby('Sentiment')['PhraseId'].agg('count').reset_index()


# <h2> On Overlapping sentences in both Train and Test set </h2>

# In[ ]:


# Lets run a simple set style intersection between the train and test dataframes to see if there are any common phrases, if so extract the phrase along with the sentiment.
# Finally we can stitch it with the predicted output ?

save_test = pd.merge(test_data_raw, train_data_raw[["Phrase", "Sentiment"]], on="Phrase", how="inner")
print ("Number of overlapping phrases  ", save_test.shape[0])
print ("% of neutral sentiment phrases",save_test[(save_test['Sentiment'] == 2)].count()[0] /save_test.shape[0])


# 

# **Word embeddings I tried **
# 
# * Pretrained word embeddings FastText, Glove, Word2Vec
# * Custom / Learned word embeddings using embedding layer
# * Custom / Learned word embeddings using Gensim implementation of FastText and Word2Vec
# 
# NN Architectures I tried:
# 
# **Word embedding based** 
# 
# * BiLSTM + Conv1D based on this  [post](http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/)
# * Conv1D + BiLSTM
# * Conv1D + BiGRU
# * Multi channel Conv1D net or parallel ConV1D [based on this post](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/).
# * Heirarchical Conv1D
# * Char-Conv1D - [based on this paper](https://arxiv.org/abs/1509.01626) and this [implementation](https://github.com/chaitjo/character-level-cnn) 
# 
# **Language modelling based** 
# * ELMo based on this [post](https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb)
# * Universal Sentence Encoder based on this [paper](https://arxiv.org/abs/1803.11175) and this [post](https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)
# 
# Another useful post for your reference on[ Word embedding vs Language Modelling](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)

# In[ ]:


word_index = tokenizer.word_index
embeddings_index = {}
embedding_size = 300
EMBEDDING_FILE =  '../input/fasttext/crawl-300d-2M.vec'

with open(EMBEDDING_FILE, 'r') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
num_words = min(vocab_size, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_size))
for word, i in word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
logging.info('Done building embedding matrix from FastText')        


# In[ ]:


from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import  GlobalMaxPool1D, SpatialDropout1D
from keras.layers import Bidirectional
from keras.models import Model
from keras import optimizers

early_stop = EarlyStopping(monitor = "val_loss", mode="min", patience = 3, verbose=1)

print("Building layers")        
nb_epoch = 25
print('starting to stitch and compile  model')

# Embedding layer for text inputs
input_words = Input((max_len, ))
x_words = Embedding(num_words, embedding_size,weights=[embedding_matrix],trainable=False)(input_words)
x_words = SpatialDropout1D(0.3)(x_words)
x_words = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x_words)
x_words = Dropout(0.2)(x_words)
x_words = Conv1D(128, 1, strides = 1,  padding='causal', activation='relu', )(x_words)
x_words = Conv1D(256, 3, strides = 1,  padding='causal', activation='relu', )(x_words)
x_words = Conv1D(512, 5, strides = 1,   padding='causal', activation='relu', )(x_words)
x_words = GlobalMaxPool1D()(x_words)
x_words = Dropout(0.2)(x_words)

x = Dense(50, activation="relu")(x_words)
x = Dropout(0.2)(x)
predictions = Dense(5, activation="softmax")(x)

model = Model(inputs=[input_words], outputs=predictions)
model.compile(optimizer='rmsprop' ,loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


print("OOV word count:", len(set(word_index) - set(embeddings_index)))


# Quick note on network/hyperparameter tuning
# 
# *  Dilated Padding with i.e 'causal' worked best instead of values **same or valid.**
# *  Word dropout after Embedding layer improved accuracy a bit
# *  nadam pushed the accuracy a bit
# *  Kernel Intilialisers didn't help, tried to change the default for CNN's
# *  BiLSTM + CNN1D performed better than CNN1D + BiLSTM
# *  BiLSTM performed better than BiGRU
# * Larger batch sizes performed better, Early stopping helped curtail overfitting.
# 
# 
# 

# In[ ]:


#fit the model
history = model.fit([X_train_words], X_train_target_binary, epochs=nb_epoch, verbose=1, batch_size = 1024, callbacks=[early_stop], validation_split = 0.2, shuffle=True)
#history = model.fit(X_train_words, X_train_target_binary, epochs=10, verbose=1, batch_size = 512,  validation_split = 0.1, shuffle=True)
train_loss = np.mean(history.history['loss'])
val_loss = np.mean(history.history['val_loss'])
print('Train loss: %f' % (train_loss*100))
print('Validation loss: %f' % (val_loss*100))


# In[ ]:


pred_test = model.predict([X_test_words], batch_size=1024, verbose = 0)
print (pred_test.shape) 
max_pred = np.round(np.argmax(pred_test, axis=1)).astype(int)
submission = pd.DataFrame({'PhraseId':test_data_raw['PhraseId'],'Sentiment': max_pred})

save_test = save_test[save_test["Sentiment"].notnull()]
save_test.drop(['SentenceId', 'Phrase'], axis=1,inplace=True)

submission =pd.merge(submission, save_test, on='PhraseId', how='left')

# This shows how poorly skilled I am with Pandas :-), I am sure many can do this couple of statements.

import math
def get_sentiment(row):
    old_s = row['Sentiment_x']
    new_s = row['Sentiment_y']
    if math.isnan(new_s):
        return int(old_s)
    else:
        return int(new_s)
    
submission["Sentiment"] = submission.apply(get_sentiment, axis=1)
submission.drop(['Sentiment_x', 'Sentiment_y'], axis=1,inplace=True)
submission["Sentiment"] = submission["Sentiment"].astype(int)
submission.to_csv('submission.csv',index=False)

