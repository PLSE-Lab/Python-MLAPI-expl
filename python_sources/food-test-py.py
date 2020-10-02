### This is a script for testing ##
## Training is done in a sepate notebook ##
## Training weights and Models are saved in RNN-LSTM Data ##

import re
import nltk

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K

import seaborn as sns
import matplotlib.pyplot as plt

def review_to_wordlist( review, remove_stopwords=True):
    #Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))
    # 5. Return a list of words
    return(b)


#load Models
## Test Time
import pickle
with open('../input/tokenizer-2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

from keras.models import model_from_json
# load json and create model
json_file = open('../input/model-cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../input/weights-improvement-cnn-09-0.61.hdf5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model 2
with open('../input/tokenizer.pickle', 'rb') as handle:
    tokenizer_rnn = pickle.load(handle)
# load json and create model
json_file = open('../input/model-rnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_rnn = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_rnn.load_weights("../input/weights-improvement-rnn-08-0.60.hdf5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def predict_rating_confidence(string):
    clean_string = []
    clean_string.append(" ".join(review_to_wordlist(str(string))))
    maxlen = 80
    sequences_test_rnn = tokenizer_rnn.texts_to_sequences(clean_string)
    string_rnn = sequence.pad_sequences(sequences_test_rnn, maxlen=maxlen)
    sequences_test_cnn = tokenizer.texts_to_sequences(clean_string)
    string_cnn = sequence.pad_sequences(sequences_test_cnn, maxlen=maxlen)
    pred_rnn = loaded_model_rnn.predict(string_rnn, verbose=1)
    pred_cnn = loaded_model.predict(string_cnn, verbose = 1)
    ensemble = (pred_cnn + pred_rnn)/2
    
    rating = np.argmax(ensemble, axis=1) + 1
    confidence = np.max(ensemble)
    return rating[0], confidence


## Test
s = '''
I have bought several of the Vitality canned dog food products and have found them all to be of good quality. 
The product looks more like a stew than a processed meat and it smells better. 
My Labrador is finicky and she appreciates this product better than  most.
'''


r, c = predict_rating_confidence(s)

print('Rating: {}, Confidence: {}'.format(r,c))
    

