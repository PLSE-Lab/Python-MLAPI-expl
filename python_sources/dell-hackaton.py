#!/usr/bin/env python
# coding: utf-8

# # Dell hackaton
# ## Team: Os Remotos
# ### Fabricio, Felipe e Heraldo

# 

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')
import multiprocessing


# In[ ]:


import os

print(os.listdir("../"))
print(os.listdir("../input/"))
debug=False

os.mkdir("../testpred")
os.mkdir("../testpred/sub")
os.mkdir("../trainpred")
os.mkdir("../trainpred/sub")
os.mkdir("../model")
os.mkdir("../model/final")


# In[ ]:


########################################
## import packages
########################################
import os
import re
import csv
import sys
from datetime import datetime
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalMaxPooling1D,GlobalAveragePooling1D ,Conv1D, MaxPooling1D, GRU,CuDNNLSTM,CuDNNGRU, Reshape, MaxPooling1D,AveragePooling1D
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical

import colorama
from colorama import Fore


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

########################################
## set directories and parameters
########################################


from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


# In[ ]:


path = '../'
TRAIN_DATA_FILE=path+'input/customerreviews/train.csv'
#TEST_DATA_FILE=path+'input/customerreviews/test1_generic_reviews.csv'
#SUB_DATA_FILE=path+'Test1_generic_reviews.csv'
TEST_DATA_FILE=path+'input/customerreviews/test2_dell_reviews.csv'
SUB_DATA_FILE=path+'Test2_dell_submission.csv'
maxlen = 200   # Maximum Sequence Size 
max_features = 250000 # Maximum Number of Words in Dictionary
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300 #300
num_dense = 256 # 256
rate_drop_lstm = 0.05
rate_drop_dense = rate_drop_lstm

loss="val_acc"
#loss="val_loss"
opt='rmsprop'
#opt='adam'

lr=0.01
from keras.optimizers import RMSprop, SGD, Nadam, Adamax, Adam
#opto = SGD(lr=lr, clipvalue=0.5)
#opto=  Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opto = RMSprop (lr=lr)
#opto = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#opto = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
##
 
act = 'relu'
Trainable=True


# In[ ]:


from contextlib import contextmanager
from datetime import datetime
@contextmanager
def timer(name):
    start_time = datetime.now()
    print(f'[{name}] Started : '+start_time.strftime("%d-%m-%Y %H:%M"))
    yield
    thour, temp_sec = divmod( (datetime.now() - start_time).total_seconds(), 3600)
    tmin, tsec = divmod(temp_sec, 60)
    print(f'[{name}] Done in :', end="");
    print(' %i h %i m and %s seconds.' % (thour, tmin, round(tsec, 2)), end="");
    print(f' Ended : '+datetime.now().strftime("%d-%m-%Y %H:%M"))


# In[ ]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    # else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.draw()
    return


# In[ ]:


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    scores = []
    for i in range(0, columns):
        scr=roc_auc_score(y_true[:, i], y_pred[:, i])
        print ("Class:",list_classes[i]," -roc:{:.5f}".format(scr))
        scores.append(scr)
    return np.array(scores).mean()


# In[ ]:


def multi_confusion(y_true, y_pred):

    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    scores = []
    for i in range(0, columns):
        rounded_predictions = np.round(y_pred[:, i], 0) 
        scr=roc_auc_score(y_true[:, i], y_pred[:, i])
        cm = confusion_matrix(y_true[:, i], rounded_predictions)
        cm_plot_labels = ['NO-'+list_classes[i],list_classes[i]]
        plot_confusion_matrix(cm, cm_plot_labels, title='Confusion '+list_classes[i]+" -roc:{:.5f}".format(scr))
    return 


# In[ ]:


import re
def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


# In[ ]:


def ReplaceThreeOrMore(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
    return pattern.sub(r"\1", s)


# In[ ]:


def splitstring(s):
    # searching the number of characters to split on
    proposed_pattern = s[0]
    for i, c in enumerate(s[1:], 1):
        if c != " ":
            if proposed_pattern == s[i:(i+len(proposed_pattern))]:
                # found it
                break
            else:
                proposed_pattern += c
    else:
        exit(1)

    return proposed_pattern


# In[ ]:


########################################
## process texts in datasets
########################################

#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text,to_lower=False, rem_urls=False, rem_3plus=False,                      split_repeated=False, rem_special=False, rep_num=False,
                     man_adj=True, rem_stopwords=False, stem_snowball=False,\
                     stem_porter=False, lemmatize=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    if rem_urls:
        text = remove_urls(text)
    if to_lower:    
        text = text.lower()
    if rem_3plus:    
        text = ReplaceThreeOrMore(text)

    if man_adj: 
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    # split them into a list
    text = text.split()
    
    if split_repeated:
        for i, c in enumerate(text):
            text[i]=splitstring(c)
    
    # Optionally, remove stop words
    if rem_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    if rem_special: 
        text=special_character_removal.sub('',text)
    
    #Replace Numbers
    if rep_num:     
        text=replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_snowball:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    if stem_porter:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in text.split()])
        
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])   
 
    # Return a list of words
    return(text)




# In[ ]:



if debug:
    train_df = pd.read_csv(TRAIN_DATA_FILE)[:1000]
    test_df = pd.read_csv(TEST_DATA_FILE)[:200]
else:
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)


# In[ ]:


# Prepare Data 
list_sentences_train = train_df["ReviewText"].fillna("NA").values
list_classes = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]
num_classes=5
#y = train_df[list_classes].values
target=train_df['Rating'].values
y1=to_categorical(target)
y=np.delete(y1, 0, axis=1)
list_sentences_test = test_df["ReviewText"].fillna("NA").values
yaux=y[:,[0]]


# In[ ]:


comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))
    


# In[ ]:



test_comments=[]
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))


# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)


# In[ ]:


print(len(sequences), 'train sequences')
print(len(test_sequences), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, test_sequences)), dtype=int)))


# In[ ]:


print(len(sequences), 'train sequences')
print(len(test_sequences), 'test sequences')
print('Max train sequence length: {}'.format(np.max(list(map(len, sequences)))))
print('Max test sequence length: {}'.format(np.max(list(map(len, test_sequences)))))


# In[ ]:


print(len(sequences), 'train sequences')
print(len(test_sequences), 'test sequences')
print('Min train sequence length: {}'.format(np.min(list(map(len, sequences)))))
print('Min test sequence length: {}'.format(np.min(list(map(len, test_sequences)))))


# In[ ]:


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=maxlen)
print('Shape of test_data tensor:', test_data.shape)


# In[ ]:


del comments
del test_sequences
del sequences
del list_sentences_train
del list_sentences_test
del train_df
#del test_df


# In[ ]:


########################################
## index GLOVE  word vectors
########################################
EMBEDDING_FILE=path+'input/glove840b300dtxt/glove.840B.300d.txt'
et="GLOVE-840B"

#EMBEDDING_FILE=path+'glove/glove.6B.300d.txt'
#et="GLOVE-6B"

#EMBEDDING_FILE= path+'prodata/toxic_clean_300d.txt'
#et='TOXIC-TXT'

#EMBEDDING_FILE= path+'fasttext/crawl-300d-2M.vec'
#et="FASTTEXT"


print('Indexing '+et+' vectors')
print("Vector",EMBEDDING_FILE )
#Glove Vectors

embeddings_index = {}
f = open(EMBEDDING_FILE,  encoding='utf8')
for line in f:
    values = line.split()
    word = ' '.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    #word = values[0]
    #coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


# In[ ]:


#########
## Glove
#########
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

########################################
## GLOVE prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(max_features, len(word_index))+1
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
#embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
nl=0
gd=0
for word, i in word_index.items():
    if i >= max_features:
        #print ('Over: ',word)
        nl = nl +1 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        gd=gd+1
    else:
        #print (word)
        nl = nl +1 

         
del embeddings_index


# In[ ]:


#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)) 
print( "Matrix", embedding_matrix.shape)
print( "Tamanho Vocabulario", len(word_index), "Maximo de Features",max_features)
print('Null word embeddings: %d' % nl)
print('Good word embeddings: %d' % gd)


# In[ ]:


def create_model0():
   
   mdln="00-max-d-dr-d-dr-d-dr"
   comment_input = Input((maxlen,))

   comment_emb = Embedding(max_features, EMBEDDING_DIM, input_length=maxlen, 
                           embeddings_initializer="uniform")(comment_input)


   m = GlobalMaxPooling1D()(comment_emb)
   d = Dense(1024, activation=act)(m)
   d = Dropout(rate_drop_dense)(d)
   d = Dense(512, activation=act)(d)
   d = Dropout(rate_drop_dense)(d)
   d = Dense(256, activation=act)(d)
   d = Dropout(rate_drop_dense)(d)
   d = Dense(128, activation=act)(d)
   d = Dropout(rate_drop_dense)(d)
   # We project onto a six-unit output layer, and squash it with a sigmoid:
   output = Dense(num_classes, activation='sigmoid')(d)

   model = Model(inputs=comment_input, outputs=output)

   model.compile(loss='binary_crossentropy',
                 optimizer=opt,
                 metrics=['accuracy'])
   
   return model,mdln


# In[ ]:


# ## Modelo LSTM Base Line 
########################################
def create_model1():

    mdln="01-lstm-att-d-b"
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=Trainable)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)

    comment_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences= embedding_layer(comment_input)
    x = lstm_layer(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(maxlen)(x)
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(num_classes, activation='sigmoid')(merged)

    model = Model(inputs=[comment_input],             outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
   
    return model, mdln


# In[ ]:


########################################
## BASE Model
########################################
def create_model2( ):

    mdln="02-base-bilstm-max-dd"
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=Trainable)

    sequence_input = Input(shape=(maxlen,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(num_dense, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_lstm))(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
   
    return model, mdln


# In[ ]:


########################################
## Modelo CONV - https://www.kaggle.com/cdubuz/keras-cnn-rnn-0-051-lb
########################################

def create_model3():

        
    
        mdln="03-conv-max-conv-max-gru-d"
        embedding_layer = Embedding(nb_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=Trainable)

        sequence_input = Input(shape=(maxlen,))
        embedded_sequences = embedding_layer(sequence_input)

        main = Dropout(rate_drop_dense)(embedded_sequences)
        main = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = GRU(64)(main)
        main = Dense(32, activation="relu")(main)
        main = Dense(num_classes, activation="sigmoid")(main)
        model = Model(inputs=sequence_input, outputs=main)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        return model, mdln

  


# In[ ]:


from keras.models import Sequential

def create_model4():

    mdln="04-conv-max-conv-max-bulstm-max-dd"


    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=Trainable)

    comment_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences= embedding_layer(comment_input)
    x = Dropout(0.2)(embedded_sequences)
    x = Conv1D(filters=EMBEDDING_DIM, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=EMBEDDING_DIM, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(num_dense, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_lstm))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=[comment_input],             outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    return model, mdln


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def create_model5():
    et='NONE'
    mdln="05-tridense"
    model = Sequential()
    sequence_input = Input(shape=(maxlen,))
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, input_dim=maxlen, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model, mdln


# In[ ]:


def create_model6( ):

    mdln="06-bicugru-con-max-avg-d"

    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)


    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
  

    tower_1 = GlobalMaxPool1D()(x)
    tower_2 = GlobalAveragePooling1D()(x)
    
    output = concatenate([  tower_1, tower_2])

    x = Dense(num_dense, activation="relu")(output)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)                         

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

   
    return model, mdln


# In[ ]:


def create_model7():
    et='NONE'
    mdln="07-biconv-mas-gru-d"
    embed_size = 256
    inp = Input(shape=(maxlen, ))
    main = Embedding(max_features, embed_size)(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(num_classes, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  
    return model, mdln


# In[ ]:


def create_model8( ):

    mdln="08-bicugru-max-dd"
    embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=Trainable)

    input_layer = Input(shape=(maxlen,))
    embedded_sequences = embedding_layer(input_layer)
    x = Bidirectional(CuDNNGRU(num_dense, return_sequences=True))(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)                         

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model, mdln


# In[ ]:


def create_model9( ):

        mdln="09-bi-lstm-max-dd"
        embedding_layer = Embedding(nb_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=Trainable)

        sequence_input = Input(shape=(maxlen,))
        embedded_sequences = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(num_dense, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_lstm))(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        preds = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model, mdln  
        
        


# In[ ]:


def create_model10( ):

    
    mdln="10-bibi-cugru-dbdd"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    main_layer = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    main_layer = Dropout(rate_drop_lstm)(main_layer)
    main_layer = Bidirectional(CuDNNGRU(num_lstm, return_sequences=False))(main_layer)
    main_layer = Dense(num_dense, activation="relu")(main_layer)
    main_layer = BatchNormalization()(main_layer)
    main_layer = Dense(64, activation="relu")(main_layer)
    main_layer = Dropout(rate_drop_dense)(main_layer)
    main_layer = Dense(32, activation="relu")(main_layer)
    main_layer = Dropout(rate_drop_dense)(main_layer)
    preds    = Dense(num_classes, activation="sigmoid")(main_layer)
    
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model, mdln  
    


# In[ ]:


def create_model11( ):

    mdln="11-bicgru-dr-bicgru-dr-d-dr"

    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)


    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    x = Dropout(rate_drop_lstm)(x)
    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=False))(x)

    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)                         

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, mdln    
    


# In[ ]:


def create_model12():

    mdln="12-bicgru-con-conv-max-avg-d-dr"

    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)


    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    
    tower_1 = Conv1D(filters=num_lstm, kernel_size=2, padding='same', activation='relu')(x)
    tower_1 = GlobalMaxPool1D()(tower_1)
    tower_2 = Conv1D(filters=num_lstm, kernel_size=2, padding='same', activation='relu')(x)
    tower_2 = GlobalAveragePooling1D()(tower_2)
    
    output = concatenate([  tower_1, tower_2])
    out = Dense(num_dense, activation="relu")(output)
    out = Dropout(rate_drop_dense)(out)
    preds = Dense(num_classes, activation="sigmoid")(out)                         

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model, mdln    
    


# In[ ]:


# ## Modelo LSTM Base Line 
########################################
def create_model13():
        et='NONE'
        mdln="13-dr-conv-max-conv-max-gru-dr"
        model = Sequential()
        inp = Input(shape=(maxlen,))
        main = Embedding(max_features, EMBEDDING_DIM)(inp)
        main = Dropout(0.2)(main)
        main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = GRU(32)(main)
        main = Dense(16, activation="relu")(main)
        main = Dense(num_classes, activation="sigmoid")(main)
        model = Model(inputs=inp, outputs=main)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model, mdln


# In[ ]:


def create_model14():

        mdln="14-culstm-sdr-bn-d-dr"
        
        comment_input = Input((maxlen,))

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        comment_emb =Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(comment_input)

        # we add a GlobalMaxPool1D, which will extract information from the embeddings
        # of all words in the document
        x = CuDNNLSTM(num_dense, return_sequences=True)(comment_emb)
        comment_emb = SpatialDropout1D(0.25)(x)
        max_emb = GlobalMaxPool1D()(comment_emb)

        # normalized dense layer followed by dropout
        main = BatchNormalization()(max_emb)
        main = Dense(64)(main)
        main = Dropout(0.5)(main)

        # We project onto a six-unit output layer, and squash it with sigmoids:
        output = Dense(num_classes, activation='sigmoid')(main)

        model = Model(inputs=comment_input, outputs=output)

        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
                                         
        
        return model, mdln


# In[ ]:


def create_model15():
        mdln="15-sdr-bigru-con-max-avg"
        inp = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(inp)
   
        x = SpatialDropout1D(rate_drop_dense)(embedding_layer)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)        
        conc = concatenate([avg_pool, max_pool])
        x = Dense(num_dense, activation="relu")(conc)
        x = Dropout(rate_drop_dense)(x)
        outp = Dense(num_classes, activation="sigmoid")(x)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
     
        return model, mdln


# In[ ]:


########################################
## BASE Model
########################################
def create_model16( ):

    mdln="16-culstm-max-d-dr"
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=Trainable)

    sequence_input = Input(shape=(maxlen,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(CuDNNLSTM(num_dense, return_sequences=True))(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss='binary_crossentropy',
            optimizer=opt,metrics=['accuracy'])
    
    return model, mdln



# In[ ]:


def create_model17( ):
    
        mdln="17-biclstm-max-d-dr-d-dr"
        embedding_layer = Embedding(nb_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=Trainable)

        sequence_input = Input(shape=(maxlen,))
        embedded_sequences = embedding_layer(sequence_input)
        x = Bidirectional(CuDNNLSTM(num_dense, return_sequences=True))(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        preds = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model, mdln  
        


# In[ ]:


def create_model18( ):
    
        mdln="18-dr-bicgru-max-d-dr"
        embedding_layer = Embedding(nb_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=Trainable)

        sequence_input = Input(shape=(maxlen,))
        embedded_sequences = embedding_layer(sequence_input)
        x = Dropout(rate_drop_dense)(embedded_sequences)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        preds = Dense(num_classes, activation="sigmoid")(x)
        model = Model(inputs=sequence_input, outputs=preds)

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model, mdln  


# In[ ]:


def create_model19( ):
    
    
        mdln="19-sdr-bicgru-con-conv-max-avg-d-dr"
        
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=Trainable)(input_layer)

        x = SpatialDropout1D(0.2)(embedding_layer)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)

        #x = Conv1D(filters=num_lstm, kernel_size=2, padding='same', activation='relu')(x)
        x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
        tower_1 = GlobalMaxPool1D()(x)
        tower_2 = GlobalAveragePooling1D()(x)

        output = concatenate([  tower_1, tower_2])

        out = Dense(num_dense, activation="relu")(output)
        out = Dropout(rate_drop_dense)(out)
        preds = Dense(num_classes, activation="sigmoid")(out)                         

        model = Model(inputs=input_layer, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model, mdln    
        
       
        
        


# In[ ]:


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


# In[ ]:


from keras import backend as K
from keras.layers import Dense


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
         for v in layer.__dict__:
             v_arg = getattr(layer,v)
             if (v != "embeddings"):
                 if hasattr(v_arg,'initializer'):
                     initializer_method = getattr(v_arg, 'initializer')
                     initializer_method.run(session=session)
                     print('reinitializing layer {}.{}'.format(layer.name, v))
             else :
                  print('keeping layer {}.{}'.format(layer.name, v)) 
            
                        
    print ("reinitializing layers...")


# In[ ]:


def evaluate_model ( label, X_train, X_valid, Y_train, Y_valid, STAMP):

            modelx=None
            modelx,mdln = create_model(label)
           

            bst_model_path=path+ "model/"+"{val_acc:.5f}-{epoch:02d}-{val_loss:.5f}_"+ STAMP + '.h5'
            #print(bst_model_path)

            #early_stopping =EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
            early_stopping =EarlyStopping(monitor=loss, patience=patience, verbose=1)
           

            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=False, save_weights_only=False, mode='auto')

            callbacks = [ early_stopping, model_checkpoint]

            
            hist=modelx.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                  shuffle=True, verbose=0, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)
            
            index_val_loss = hist.history['val_loss'].index(min(hist.history['val_loss']))
            bst_model_path=path+ "model/"+"{:.5f}-".format(hist.history['val_acc'][index_val_loss])+"{:02d}".format(index_val_loss+1)+"-{:.5f}".format(hist.history['val_loss'][index_val_loss])+"_"+STAMP+'.h5'
 
            #reset_weights(model)
            K.clear_session()
            modelx=None
            del modelx
            return bst_model_path


# In[ ]:


def run_cross_validation_create_models(label,nsplits=10,epochs=3,patience=3):
        random_state = 999

        yfull_train = dict()
        train_full=  []
        test_full = []

        kf = skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
        num_fold = 0
        sum_score = 0
        bestmodel,mdln=create_model(label)
        print (Fore.GREEN+"\n"+mdln," ***********************************************************")
        print (bestmodel.summary() )
        print (Fore.BLACK)
        saved_models = []
        score = np.zeros(nsplits)
        score_partial = np.zeros(nsplits)
        for i, (train_index, test_index) in enumerate(kf.split(data,yaux)):
            num_fold += 1
            print('Fold:',num_fold)

            X_train, X_valid = data[train_index],data[test_index]
            Y_train, Y_valid = y[train_index], y[test_index]
            print('Start KFold number {} from {} - Fitting'.format(num_fold, nsplits))

           
            bst_model_path=evaluate_model (label,X_train, X_valid, Y_train, Y_valid, STAMP)
            
          
            print ("Validating")

            ### Getting the Best Model
            bestmodel,mdln=create_model(label)
            bestmodel.load_weights(bst_model_path)          
            
            #predictions_valid = bestmodel.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
            predictions_valid = bestmodel.predict(X_valid, batch_size=batch_size, verbose=0)

            score_partial[i] = multi_roc_auc_score(Y_valid, predictions_valid)
            print(Fore.BLUE +'Partial Score roc_auc: {:.5f}\n'.format(score_partial[i])+Fore.BLACK)
            info_string = '{:.5f}'.format(score_partial[i]) +"_fl_"+'{:02d}'.format(num_fold) +"_"+STAMP
 

            train_pred = bestmodel.predict(data, batch_size=batch_size, verbose=1)
            score[i] = multi_roc_auc_score(y, train_pred)
            print(Fore.GREEN +'Full Score roc_auc: {:.5f}\n'.format(score[i])+Fore.BLACK)
            train_full.append(train_pred)
            
            test_pred = bestmodel.predict(test_data, batch_size=batch_size, verbose=1)
            test_full.append(test_pred)
                       
            #newfile= path+"model/"+"{:.5f}-roc-".format(score_partial[i])+bst_model_path[-(len(STAMP)+21):]
            newfile= path+"model/final/"+"{:.5f}-roc-".format(score_partial[i])+bst_model_path[35:]
            os.rename(bst_model_path, newfile)
            
            sum_score += score_partial[i]*len(test_index)

            saved_models.append(bestmodel)

            del bestmodel
        scoreF = sum_score/len(data)
        
        train_res = np.array( merge_several_folds_mean(train_full, nsplits))  
        
        print(Fore.RED +'roc_uac train independent: {:.5f}\n'.format(scoreF))
        print ("Teste Internal Score {:.5f}".format(score_partial.mean()) )
        print ("Teste External Score {:.5f}\n".format(score.mean()) )
            
        test_res  = merge_several_folds_mean(test_full, nsplits)
        results=np.array(test_res)

        ratings=np.argmax(results, axis=1)
        print(ratings)

        submit_df = pd.DataFrame({"ReviewText": test_df["ReviewText"], "Rating": ratings})
        submit_df.to_csv('submission1.csv')
        print ("Submit Rows:",submit_df["Rating"].count())    
            
        print(Fore.BLACK)
          
        K.clear_session()   

        return info_string, saved_models 
        


# In[ ]:


def create_model (label):
    
            if (label=="0"):
                return create_model0()
            if (label=="1"):
                return create_model1()
            if (label=="2"):
                return create_model2()
            if (label=="3"):
                return create_model3()            
            if (label=="4"):
                return create_model4()
            if (label=="5"):
                return create_model5()
            if (label=="6"):
                return create_model6()
            if (label=="7"):
                return create_model7()  
            if (label=="8"):
                return create_model8()
            if (label=="9"):
                return create_model9() 
            if (label=="10"):
                return create_model10()            
            if (label=="11"):
                return create_model11()                
            if (label=="12"):
                return create_model12()                
            if (label=="13"):
                return create_model13()                
            if (label=="14"):
                return create_model14()                
            if (label=="15"):
                return create_model15()                
            if (label=="16"):
                return create_model16()   
            if (label=="17"):
                return create_model17() 
            if (label=="18"):
                return create_model18() 
            if (label=="19"):
                return create_model19() 
            return None,"None"
            
            
            
            
            
            


# In[ ]:


nsplits = 5
epochs=5
patience=1
trainpred=True
testpred=True
batch_size=512
STAMP="AA"
best_models= ["15","16"]

for label in  ["16"]:
       with timer(label):
             info_string, modelos = run_cross_validation_create_models(label, nsplits,epochs, patience)



