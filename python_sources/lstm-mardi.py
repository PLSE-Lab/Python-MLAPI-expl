import gensim, logging, os, pickle, multiprocessing, random, collections, unicodedata
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import keras.backend as K
import tensorflow as tf

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from string import punctuation

from gensim.models.word2vec import Word2Vec
from emoji.unicode_codes import UNICODE_EMOJI

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.optimizers import Adam
##############################################################################################
corpus = []
labels = []
        
paths_non_treated = ["../input/data-sets/Disneyland_Orlando_trip_advisor_train.csv",
                    "../input/data-sets/Disneyland_Paris_trip_advisor_train_4000.csv",
                    "../input/data-sets/Disneyland_Paris_trip_advisor_train_4000_8000.csv",
                    "../input/data-sets/Disneyland_Paris_trip_advisor_train_8000_11400.csv",
                    "../input/data-sets/Parc_Disneyland_trip_advisor_train.csv"]

# Importer le module nltk pour l'analyse de texteimport nltk
import nltk
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
import re
import unidecode
from string import punctuation
from nltk.tokenize import RegexpTokenizer

# Algo de stemming permettant de supprimer automatiquement les suffixes pour n'obtenir que la racine des mots. On parle de racinisation.
stemmer = nltk.stem.SnowballStemmer('french')

# Invoquer des mots vides en francais et les stocker dans des variables
stopWords = stopwords.words('french')
stopWords.remove(stemmer.stem("ne"))
stopWords.remove(stemmer.stem("pas"))
stopWords.remove(stemmer.stem("mais"))
stopWords.append("le")
stopWords.append("la")

tokenizer = RegexpTokenizer(r'\w+')
                    
import csv
for path in paths_non_treated:
    f = open(path, 'rt')
    reader = csv.reader(f, delimiter='µ')
    m = 0
    for row in reader:
        if m > 0:
            line = row[0]
            line = line.replace("\n", ".")
            for i in range(10):
                line = line.replace(str(i), " ")
            line = line.split(".")
            for l in range(len(line)):
                tokens = tokenizer.tokenize(line[l])
                reviewProcessed = ''
                for token in tokens:
                    if token.lower() not in stopWords:
                        token = stemmer.stem(token)
                        reviewProcessed += token + " "
                if reviewProcessed != "" and reviewProcessed != "." and reviewProcessed != " " and reviewProcessed != "plus ":
                    corpus.append(reviewProcessed)
                    labels.append(row[1])
        m = m + 1

import random

augmented_corpus = []
augmented_labels =[]

for t in [t for t in range(len(labels)) if (labels[t] == "-1" or labels[t] == "0")]:
    tok =  word_tokenize(corpus[t])
    indices = np.arange(len(tok))
    for q in range(round(len(indices)/10)):
        np.random.shuffle(indices)
        res = ""
        res = " ".join([str(tok[j])+" " for j in indices])
        augmented_corpus.append(res)
        augmented_labels.append(labels[t])

corpus = corpus + augmented_corpus
labels = labels + augmented_labels

c = list(zip(corpus, labels))
random.shuffle(c)
corpus, labels = zip(*c)

import collections
counter=collections.Counter(labels)
print(counter)
##########################################################################
import keras.backend as K
import multiprocessing
import tensorflow as tf

from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

# Set random seed (for reproducibility)
np.random.seed(1000)

# Select whether using Keras with or without GPU support
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)


print('Corpus size: {}'.format(len(corpus)))

# Gensim Word2Vec model
vector_size = 256
window_size = 10

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [t.lower() for t in word_tokenize(tweet)]
    tokenized_corpus.append(tokens)

# Create Word2Vec
word2vec = Word2Vec(sentences=tokenized_corpus,
                    min_count=10,
                    size=vector_size,
                    window=window_size,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())
 
# Copy word vectors and delete Word2Vec model  and original corpus to save memory
X_vecs = word2vec.wv
word2vec.wv.save_word2vec_format('Word2vec_model.bin')
#X_vecs = Word2Vec.load('model.bin')

stemmer = nltk.stem.SnowballStemmer('french')
# view similar words based on gensim's model
similar_words = {search_term: [item[0] for item in word2vec.wv.most_similar([search_term], topn=10)]
                  for search_term in [stemmer.stem("montagne")]}
print(similar_words)

#del word2vec
#del corpus

# Compute average and max tweet length

select = []
nbre_mots = []
aug_split_tweets = []
aug_split_labels = []

i=0
for tweet in tokenized_corpus:
    if len(tweet) > 30 :
        for f in range(0, len(tweet), 30):
            aug_split_tweets.append(tweet[f:(f+30)])
            aug_split_labels.append(labels[i])
    else :
        select.append(int(i))
    i+=1

    
labels_selected = []
tokenized_corpus_selected = []
labels_selected = [labels[t] for t in select]
tokenized_corpus_selected = [tokenized_corpus[t] for t in select]

labels_selected.append(aug_split_tweets)
tokenized_corpus_selected.append(aug_split_labels)

# Train subset size (0 < size < len(tokenized_corpus))
train_size = round(len(tokenized_corpus_selected) * 8 / 10)

# Test subset size (0 < size < len(tokenized_corpus) - train_size)
test_size = len(tokenized_corpus_selected) - train_size

# Tweet max length (number of tokens)
max_tweet_length = 30

# Create train and test sets
# Generate random indexes
indexes = set(np.random.choice(len(tokenized_corpus_selected), train_size + test_size, replace=False))

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_train = np.zeros((train_size, 3), dtype=np.int32)
X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_test = np.zeros((test_size, 3), dtype=np.int32)

for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus_selected[index]):
        if t >= max_tweet_length:
            break
        if token not in X_vecs:
            continue
        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]

    if i < train_size:
        if labels_selected[index] == "-1":
            Y_train[i, :] = [1.0, 0.0, 0.0]
        elif labels_selected[index] == "0":
            Y_train[i, :] = [0.0, 1.0, 0.0]
        else :
            Y_train[i, :] = [0.0, 0.0, 1.0]
    else:
        if labels_selected[index] == "-1":
            Y_test[i - train_size, :] = [1.0, 0.0, 0.0]
        elif labels_selected[index] == "0":
            Y_test[i - train_size, :] = [0.0, 1.0, 0.0]
        else:
            Y_test[i - train_size, :] = [0.0, 0.0, 1.0]

def sample(preds, temperature=1.0):
    return np.argmax(preds)
    
#Importer les tweets pour tester le modele
import nltk
stemmer = nltk.stem.SnowballStemmer('french')
from nltk.corpus import stopwords
stopWords = stopwords.words('french')
from nltk import word_tokenize

from emoji.unicode_codes import UNICODE_EMOJI

def is_emoji(s):
    return s in UNICODE_EMOJI
    
def import_tweets(path):
    tweets = []
    labels_tweets = []
    import csv
    f = open(path, 'rt')
    reader = csv.reader(f, delimiter='µ')
    i = 0
    for i, row in enumerate(reader):
        tweet = [stemmer.stem(t.replace("#", "")) for t in word_tokenize(row[1]) if (not t in stopWords and t.replace("#", "") !="")]
        if len(tweet) > 0 and i > 0:
            for l in range(len(tweet)):
                for k in range(len(tweet[l])):
                    if is_emoji(tweet[l][k]) == True:    
                        if UNICODE_EMOJI[tweet[l][k]] in list([":smiling_cat_face_with_heart-eyes:", ":smiling_cat_face_with_open_mouth:", ":grinning_face_with_star_eyes:", 
                        ":grinning_face:", ":face_with_tears_of_joy:", ":smiling_face_with_open_mouth:", ":smiling_face_with_open_mouth_and_smiling_eyes:",
                        ":smiling_face_with_heart-shaped_eyes:", ":winking_face:"]):
                            tweet.append(stemmer.stem("aimer"))
                        elif UNICODE_EMOJI[tweet[l][k]] in [":loudly_crying_face:", ":tired_face:", ":crying_cat_face:"]:
                            tweet.append(stemmer.stem("pleurer"))
                        elif UNICODE_EMOJI[tweet[l][k]] in [":face_screaming_in_fear:", ":fearful_face:"]:
                            tweet.append(stemmer.stem("peur"))
                        elif UNICODE_EMOJI[tweet[l][k]] in [":disappointed_face:"]:
                            tweet.append(stemmer.stem("decu"))
                        elif UNICODE_EMOJI[tweet[l][k]] in [":pouting_face:", ":angry_face:", ":pouting_cat_face:"]:
                            tweet.append(stemmer.stem("enerver"))   
                        elif UNICODE_EMOJI[tweet[l][k]] in [":heavy_black_heart:", ":sparkling_heart:", ":growing_heart:", ":blue_heart:", ":green_heart:", ":yellow_heart:"]:
                            tweet.append(stemmer.stem("adorer"))   
            tweets.append(tweet)
            labels_tweets.append(row[-1])
        
    X = np.zeros((len(tweets), max_tweet_length, vector_size), dtype=K.floatx())

    for index in range(len(tweets)):
        for t, token in enumerate(tweets[index]):
            if token in X_vecs and t < max_tweet_length:
                X[index, t, :] = X_vecs[token]
    return X, labels_tweets

X, Y = import_tweets("../input/tweets/tweet_streaming.csv")

import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def get_model(X_train, Y_train, X_test, Y_test, batch_size, nb_epochs, lr):
    # Keras convolutional model
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(max_tweet_length, vector_size), return_sequences = False, go_backwards = True))

    model.add(Dense(128, activation='tanh'))

    model.add(Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])
      
    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs,
              validation_data=(X_test, Y_test),
              verbose = 2)
          
    return model
    
    
model = get_model(X_train, Y_train, X_test, Y_test, 64, 100, 0.0001)

model.save('LSTM.h5')


prediction = model.predict(X)
predictions_tab = []
for t in range(len(prediction)):
    predictions_tab.append(sample(prediction[t])-1)

error = 0
for i in range(362):
    #print(str(predictions_tab[i]) + "    |    " + str(Y[i]) + "\n")
    if str(predictions_tab[i]) != str(Y[i]):
        error += 1
error = error / 362
print (error)