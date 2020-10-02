#!/usr/bin/env python
# coding: utf-8

# # Improved LSTM baseline
# 
# This kernel is a somewhat improved version of [Keras - Bidirectional LSTM baseline](https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051) along with some additional documentation of the steps. (NB: this notebook has been re-run on the new test set.)

# In[ ]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from nltk.tokenize import TweetTokenizer
from unidecode import unidecode

import gc
from keras import backend as K


# In[ ]:


import os, math, operator, csv, random, pickle,re
import tensorflow as tf
import pandas as pd
import gc

import keras
from keras.models import Model
from keras.layers import MaxPooling1D, BatchNormalization, Permute, Lambda, Activation, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Embedding, Dropout, Input, CuDNNGRU, merge, CuDNNLSTM, Flatten, TimeDistributed, concatenate, SpatialDropout1D, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pyplot as plt

import gc
from keras import backend as K

from nltk.tokenize import TweetTokenizer

from unidecode import unidecode

from sklearn.model_selection import KFold, train_test_split
from keras import backend as K

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)


# We include the GloVe word vectors in our input files. To include these in your kernel, simple click 'input files' at the top of the notebook, and search 'glove' in the 'datasets' section.

# In[ ]:


import os
os.listdir('../input/')


# In[ ]:


os.listdir('../input/glove6b50d/')


# In[ ]:


os.listdir('../input/fasttext-crawl-300d-2m/')


# In[ ]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE1=f'{path}glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'
EMBEDDING_FILE2=f'{path}fasttext-crawl-300d-2m/rawl-300d-2M.vec'
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# In[ ]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")


# Set some basic config parameters:

# In[ ]:


BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


# In[ ]:


#embed_size = 50 # how big is each word vector
embedding_dim_facebook = 300
embedding_dim_google = 300
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


# Read in our data and replace missing values:

# In[ ]:


training_samples_count = 149571
validation_samples_count = 10000

length_threshold = 20000 #We are going to truncate a comment if its length > threshold
word_count_threshold = 900 #We are going to truncate a comment if it has more words than our threshold
words_limit = 310000

#We will filter all characters except alphabet characters and some punctuation
valid_characters = " " + "@$" + "'!?-" + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
valid_characters_ext = valid_characters + "abcdefghijklmnopqrstuvwxyz".upper()
valid_set = set(x for x in valid_characters)
valid_set_ext = set(x for x in valid_characters_ext)

#List of some words that often appear in toxic comments
#Sorry about the level of toxicity in it!
toxic_words = ["poop", "crap", "prick", "twat", "wikipedia", "wiki", "hahahahaha", "lol", "bastard", "sluts", "slut", "douchebag", "douche", "blowjob", "nigga", "dumb", "jerk", "wanker", "wank", "penis", "motherfucker", "fucker", "fuk", "fucking", "fucked", "fuck", "bullshit", "shit", "stupid", "bitches", "bitch", "suck", "cunt", "dick", "cocks", "cock", "die", "kill", "gay", "jewish", "jews", "jew", "niggers", "nigger", "faggot", "fag", "asshole"]
astericks_words = [('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'), ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'), ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'), ('fu*cking', 'fucking'), ('****ing', 'fucking'), ('a**hole', 'asshole'), ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'), ('f***ing', 'fucking'), ('f**king', 'fucking'), ('f*cking', 'fucking'), ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'), ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'), ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'), ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'), ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'), ('f**ks', 'fucks'), ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'), ('cr*p', 'crap'), ('d*ck', 'dick'), ('f***', 'fuck'), ('f**k', 'fuck'), ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'), ('s***', 'shit'), ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')]
fasttext_misspelings = {"'n'balls": 'balls', "-nazi's": 'nazis', 'adminabuse': 'admin abuse', "admins's": 'admins', 'arsewipe': 'arse wipe', 'assfack': 'asshole', 'assholifity': 'asshole', 'assholivity': 'asshole', 'asshoul': 'asshole', 'asssholeee': 'asshole', 'belizeans': 'mexicans', "blowing's": 'blowing', 'bolivians': 'mexicans', 'celtofascists': 'fascists', 'censorshipmeisters': 'censor', 'chileans': 'mexicans', 'clerofascist': 'fascist', 'cowcrap': 'crap', 'crapity': 'crap', "d'idiots": 'idiots', 'deminazi': 'nazi', 'dftt': "don't feed the troll", 'dildohs': 'dildo', 'dramawhores': 'drama whores', 'edophiles': 'pedophiles', 'eurocommunist': 'communist', 'faggotkike': 'faggot', 'fantard': 'retard', 'fascismnazism': 'fascism', 'fascistisized': 'fascist', 'favremother': 'mother', 'fuxxxin': 'fucking', "g'damn": 'goddamn', 'harassmentat': 'harassment', 'harrasingme': 'harassing me', 'herfuc': 'motherfucker', 'hilterism': 'fascism', 'hitlerians': 'nazis', 'hitlerites': 'nazis', 'hubrises': 'pricks', 'idiotizing': 'idiotic', 'inadvandals': 'vandals', "jackass's": 'jackass', 'jiggabo': 'nigga', 'jizzballs': 'jizz balls', 'jmbass': 'dumbass', 'lejittament': 'legitimate', "m'igger": 'nigger', "m'iggers": 'niggers', 'motherfacking': 'motherfucker', 'motherfuckenkiwi': 'motherfucker', 'muthafuggas': 'niggas', 'nazisms': 'nazis', 'netsnipenigger': 'nigger', 'niggercock': 'nigger', 'niggerspic': 'nigger', 'nignog': 'nigga', 'niqqass': 'niggas', "non-nazi's": 'not a nazi', 'panamanians': 'mexicans', 'pedidiots': 'idiots', 'picohitlers': 'hitler', 'pidiots': 'idiots', 'poopia': 'poop', 'poopsies': 'poop', 'presumingly': 'obviously', 'propagandaanddisinformation': 'propaganda and disinformation', 'propagandaministerium': 'propaganda', 'puertoricans': 'mexicans', 'puertorricans': 'mexicans', 'pussiest': 'pussies', 'pussyitis': 'pussy', 'rayaridiculous': 'ridiculous', 'redfascists': 'fascists', 'retardzzzuuufff': 'retard', "revertin'im": 'reverting', 'scumstreona': 'scums', 'southamericans': 'mexicans', 'strasserism': 'fascism', 'stuptarded': 'retarded', "t'nonsense": 'nonsense', "threatt's": 'threat', 'titoists': 'communists', 'twatbags': 'douchebags', 'youbollocks': 'you bollocks'}
acronym_words = {} #{"btw":"by the way", "yo": "you", "u": "you", "r": "are", "ur": "your", "ima": "i am going to", "imma": "i am going to", "i'ma":"i am going to", "cos":"because", "coz":"because", "stfu": "shut the fuck up", "wat": "what"}


# In[ ]:


cont_patterns = [
    (r'(W|w)on\'t', r'will not'),
    (r'(C|c)an\'t', r'can not'),
    (r'(I|i)\'m', r'i am'),
    (r'(A|a)in\'t', r'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def split_word(word, toxic_words):
    if word == "":
        return ""
    
    lower = word.lower()
    for toxic_word in toxic_words:
        start = lower.find(toxic_word)
        if start >= 0:
            end = start + len(toxic_word)
            result = " ".join([word[0:start], word[start:end], split_word(word[end:], toxic_words)])
            return result.replace("  ", " ").strip()
    return word

tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
def word_tokenize(sentence):
    sentence = sentence.replace("$", "s")
    sentence = sentence.replace("@", "a")    
    sentence = sentence.replace("!", " ! ")
    sentence = sentence.replace("?", " ? ")
    
    return tknzr.tokenize(sentence)

def replace_url(word):
    if "http://" in word or "www." in word or "https://" in word or "wikipedia.org" in word:
        return ""
    return word

def normalize_by_dictionary(normalized_word, dictionary):
    result = []
    for word in normalized_word.split():
        if word == word.upper():
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()].upper())
            else:
                result.append(word)
        else:
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()])
            else:
                result.append(word)
    
    return " ".join(result)


# In[ ]:


from spacy.symbols import nsubj, VERB, dobj
import spacy
nlp = spacy.load('en')

def normalize_comment(comment):
    comment = unidecode(comment)
    comment = comment[:length_threshold]
    
    normalized_words = []
    
    for w in astericks_words:
        if w[0] in comment:
            comment = comment.replace(w[0], w[1])
        if w[0].upper() in comment:
            comment = comment.replace(w[0].upper(), w[1].upper())
    
    for word in word_tokenize(comment):
        #for (pattern, repl) in patterns:
        #    word = re.sub(pattern, repl, word)

        if word == "." or word == ",":
            normalized_words.append(word)
            continue
        
        word = replace_url(word)
        if word.count(".") == 1:
            word = word.replace(".", " ")
        filtered_word = "".join([x for x in word if x in valid_set])
                    
        #Kind of hack: for every word check if it has a toxic word as a part of it
        #If so, split this word by swear and non-swear part.
        normalized_word = split_word(filtered_word, toxic_words)
        #normalized_word = normalize_by_dictionary(normalized_word, hyphens_dict)
        #normalized_word = normalize_by_dictionary(normalized_word, merged_dict)
        #normalized_word = normalize_by_dictionary(normalized_word, misspellings_dict)
        normalized_word = normalize_by_dictionary(normalized_word, fasttext_misspelings)
        normalized_word = normalize_by_dictionary(normalized_word, acronym_words)

        normalized_words.append(normalized_word)
        
    normalized_comment = " ".join(normalized_words)
    
    result = []
    for word in normalized_comment.split():
        if word.upper() == word:
            result.append(word)
        else:
            result.append(word.lower())
    
    #apparently, people on wikipedia love to talk about sockpuppets :-)
    result = " ".join(result)
    if "sock puppet" in result:
        result = result.replace("sock puppet", "sockpuppet")
    
    if "SOCK PUPPET" in result:
        result = result.replace("SOCK PUPPET", "SOCKPUPPET")
    
    return result


# In[ ]:


def read_data_files(train_filepath, test_filepath):
    #read train data
    train = pd.read_csv(train_filepath)


    labels = train[categories].values
    
    #read test data
    test = pd.read_csv(test_filepath)

    test_comments = test["comment_text"].fillna("_na_").values

    #normalize comments
    np_normalize = np.vectorize(normalize_comment)
    comments = train["comment_text"].fillna("_na_").values
    normalized_comments = np_normalize(comments)
    del comments
    gc.collect()

    
    comments = test["comment_text"].fillna("_na_").values
    normalized_test_comments = np_normalize(test_comments)
    del comments
    gc.collect()
       

    print('Shape of data tensor:', normalized_comments.shape)
    print('Shape of label tensor:', labels.shape)
    print('Shape of test data tensor:', normalized_test_comments.shape)
    
    return (labels, normalized_comments, normalized_test_comments)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'labels, x_train, x_test = read_data_files(TRAIN_DATA_FILE, TEST_DATA_FILE) ')


# In[ ]:


#train = pd.read_csv(TRAIN_DATA_FILE)
#test = pd.read_csv(TEST_DATA_FILE)

#list_sentences_train = train["comment_text"].fillna("_na_").values
#list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#y = train[list_classes].values
#list_sentences_test = test["comment_text"].fillna("_na_").values


# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# In[ ]:


get_ipython().run_cell_magic('time', '', "embedding_matrix = np.load('../input/embedding-2/embedding_matrix_big.npy')")


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x_train, labels, test_size = 0.1)


# In[ ]:


def build_model(embedding_matrix):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(6, activation='sigmoid')(hidden)
    
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# In[ ]:


EPOCHS = 5
SEEDS = 10

pred = 0

for ii in range(SEEDS):
    model = build_model(embedding_matrix)
    for global_epoch in range(EPOCHS):
        print(global_epoch)
        model.fit(
                    x_train,
                    y_train,
                    validation_data = (x_valid, y_valid),
                    batch_size=128,
                    epochs=1,
                    verbose=2,
                    callbacks=[
                        LearningRateScheduler(lambda _: 1e-3 * (0.5 ** global_epoch))
                    ]
                )
        val_preds = model.predict(x_valid)
        AUC = 0
        for i in range(6):
             AUC += roc_auc_score(y_valid[:,i], val_preds[:,i])/6.
        print(AUC)

    pred += model.predict(x_test, batch_size = 1024, verbose = 1)/SEEDS
    np.save('pred', pred)
    model.save_weights('model_weights_'+str(ii)+'.h5')
    os.system('gzip '+'model_weights_'+str(ii)+'.h5')


# In[ ]:


#tokenizer = Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(list(list_sentences_train))
#list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
#X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[ ]:


#def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.

# In[ ]:


#all_embs = np.stack(embeddings_index.values())
#emb_mean,emb_std = all_embs.mean(), all_embs.std()
#emb_mean,emb_std


# In[ ]:


#word_index = tokenizer.word_index
#nb_words = min(max_features, len(word_index))
#embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#for word, i in word_index.items():
#    if i >= max_features: continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

# In[ ]:


#inp = Input(shape=(maxlen,))
#x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
#x = GlobalMaxPool1D()(x)
#x = Dense(50, activation="relu")(x)
#x = Dropout(0.1)(x)
#x = Dense(6, activation="sigmoid")(x)
#model = Model(inputs=inp, outputs=x)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Now we're ready to fit out model! Use `validation_split` when not submitting.

# In[ ]:


#model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1);


# And finally, get predictions for the test set and prepare a submission CSV:

# In[ ]:


#y_test = model.predict([X_te], batch_size=1024, verbose=1)
#sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
#sample_submission[list_classes] = y_test
#sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


submission[list_classes] = (pred)
submission.to_csv("submission.csv", index = False)


# In[ ]:




