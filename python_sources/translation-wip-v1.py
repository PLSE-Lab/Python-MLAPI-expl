#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # l|inear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tqdm
from tqdm import tqdm
from keras_tqdm import TQDMNotebookCallback
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import keras
import tensorflow as tf


# In[ ]:


from keras.models import Model,Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM 
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam ,Adagrad ,RMSprop
from keras.losses import sparse_categorical_crossentropy


# In[ ]:


train_d=pd.read_csv('../input/english-to-french/small_vocab_en.csv',sep="\t",header=None,names=['Text'])
train_d=train_d.rename(columns={'0':'En_text'})
test_d=pd.read_csv('../input/english-to-french/small_vocab_fr.csv',sep="\t",header=None,names=['Text'])
test_d=test_d.rename(columns={'0':'Fr_text'})


# In[ ]:


test_d.head(10)


# Add multiprocessing/threading to make this nigga faster
# 

# In[ ]:


def remove_stopwords(sentence,language='english'):
    """
    Parameters
    ----------
    sentence : string
        String containing text
    language : str, optional
        The language code to be used for filtering stopwords. The default is 'english'.

    Raises
    ------
    TypeError
        When wrong input to the fucntion.

    Returns
    -------
    filtered_sentence : str
        The filtered sentence

    """

    if sentence is type(str):
            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            from nltk import word_tokenize
            stop_words = stopwords.words(language)
            word_tokens = word_tokenize(sentence) 
            filtered_sentence =' '.join(map(str,[w for w in word_tokens if not w in stop_words]))
            return filtered_sentence
    else:
        raise TypeError("Expected type str")
    


# In[ ]:


def words_distribution(dataframe,topx=10,stopwords=False,stopwordslist=None):
    """
    

    Parameters
    ----------
    dataframe : pd.Dataframe
        The dataframe containing text.
    topx : int, optional
        Display the top n words in the given distribution. The default is 10.
    stopwords : bool, optional
        Stopwords in various lagnuages. The default is False.
    stopwordslist : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    TypeError
        When wrong type of iterable is entered in the given fucntion

    Returns
    -------
    None.

    """
    
    if  isinstance(dataframe,pd.DataFrame):
        if topx is not type(int):
            from nltk import FreqDist
            from nltk import word_tokenize
            sens=[sen[0] for sen in dataframe.values if sen!=']' or sen!='[']
            sens=[''.join(sen[0].lower()) for sen in dataframe.values if sen!=']' or sen!='[']
            words=word_tokenize(str([sens[i] for i in range(len(sens))]))
            freq=FreqDist(words)
            freq.plot(topx)
        else:
                 raise TypeError("Expected type int got type {0}".format(type(topx)))
                
    else:
        raise TypeError("Expected type pd.DataFrame got type {0}".format(type(dataframe)))          


# Usage defined below

# In[ ]:



#words_distribution(dataframe=test_d)


# Steps to be performed :
# **Tokenizing**
# **Paddding to prevent length issues.**

# In[ ]:


def tokenize(dataframe,char_level=False):
    """
    Parameters
    ----------
    dataframe : pd.Dataframe
        Dataframe to generate tokens
    char_level : bool
        Create character level or word level tokens. The default is False.

    Raises
    ------
    TypeError
        When character level is not type bool

    Returns
    -------
    text_sequences : list
        A list containing the tokenized vocabulary
    tk : keras_preprocessing.text.Tokenizer
       Keras tokenizer

    """
    if  isinstance(dataframe,pd.DataFrame):
        data=np.array(dataframe.values).ravel()
        if char_level is not type(bool):
                from keras.preprocessing.text import Tokenizer
                if char_level==False:
                    tk=Tokenizer(lower=True ,char_level=False)
                    tk.fit_on_texts(data)
                    text_sequences=tk.texts_to_sequences(data)
                else:
                    tk=Tokenizer(lower=True ,char_level=True)
                    tk.fit_on_texts(data)
                    text_sequences=tk.texts_to_sequences(data)  
                return text_sequences,tk
        else:
            TypeError("Expected type bool got type {0}".format(type(topx)))
    else:
        raise TypeError("Expected type pd.DataFrame got type {0}".format(type(dataframe))) 


# In[ ]:


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    from keras.preprocessing.sequence import pad_sequences
    return pad_sequences(x, maxlen=length, padding='post',value=0.0) #Pad at the end


# In[ ]:


sequences,tokenizer=tokenize(test_d,char_level=False)


# In[ ]:


len(tokenizer.word_index)


# Verifying padding

# In[ ]:


padseq=pad(sequences,10)


# Created a list of sequences with number
# 
# Creates dictionary of** * Word *:*Frequency***
# 
# **Note:0 is reserved for Padding**
# 

# In[ ]:


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


# In[ ]:


preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(train_d,test_d)


# In[ ]:


preproc_french_sentences.shape[::]


# Visualize the English vocabulary

# In[ ]:


english_tokenizer.word_index


# In[ ]:


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


# In[ ]:


padding_english_sent = pad(preproc_english_sentences,preproc_french_sentences.shape[1])
padding_english_sent= padding_english_sent.reshape((-1, preproc_french_sentences.shape[-2], 1)) #Both should be of same dimensions


# In[ ]:


padding_english_sent.shape


# In[ ]:


english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
#print("Prediction:")
#print(logits_to_text(model.predict(np.ndarray('How are you doing'), french_tokenizer)))
#padding_english_sent.shape
preproc_french_sentences.shape
#padding_english_sent[:1][0]


# In[ ]:


def gru_rnn_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):   
        model = Sequential()
        model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
        model.add(TimeDistributed(Dense(1024, activation='tanh')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

        # Compile model
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(0.0054),
                      metrics=['accuracy','sparse_categorical_crossentropy'])
        return model


# In[ ]:



model = gru_rnn_model(padding_english_sent.shape,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))
print(model.summary())


# **Trying Early Stopping**
# 
# However,has no effect since loss is high and also not overfitting as much. An overall bad model tbh.

# In[ ]:


padding_english_sent.shape


# In[ ]:


from keras.callbacks.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_accuracy',mode='max',patience=5) #Stop when model loss cannot reach a greater minmum value


# In[ ]:


history=model.fit(padding_english_sent, preproc_french_sentences, batch_size=1024, epochs=30,validation_split=0.2,callbacks=None)


# In[ ]:


def plot_model_performance(*criteria):
    for c in criteria:
            fig, axes= plt.subplots()
            axes.plot(history.history[''+c])
            axes.plot(history.history['val_'+c])
            axes.set_title('Model '+c)
            axes.set_ylabel(''+c)
            axes.set_xlabel('Epoch')
            axes.legend(['Train', 'Test'], loc='upper right')
    return axes
          


# **GRU PLOT**
# ![](http://wikimedia.org/api/rest_v1/media/math/render/svg/12dd26f9c68a2dc8aab60bb9627d9440d4e6952b)
# **Original Paper**
#                     [https://arxiv.org/pdf/1406.1078.pdf](http://)

# In[ ]:


gru=plot_model_performance('accuracy','loss')


# In[ ]:


model.save('my_model.h5')


# In[ ]:


def lstm_rnn_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):   
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    #model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(0.0054),
                  metrics=['accuracy','sparse_categorical_crossentropy'])
    return model


# In[ ]:


model = lstm_rnn_model(padding_english_sent.shape,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))
model.summary()


# In[ ]:


history=model.fit(padding_english_sent, preproc_french_sentences, batch_size=1024, epochs=30,validation_split=0.2,callbacks=[TQDMNotebookCallback()])


# **LSTM PLOT**
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2db2cba6a0d878e13932fa27ce6f3fb71ad99cf1)
# 
# Also note GRU is faster to train than LSTM due to reduced gates and also less number of Parameters(813K vs 879K)

# In[ ]:


axes=plot_model_performance('accuracy','loss')


# ** Model with Embedding**
# 

# THIS PADDING STEP IS A BIT DIFFERENT FROM  OTHER MODELS

# In[ ]:


padding_english_sent= pad(preproc_english_sentences, preproc_french_sentences.shape[1])
padding_english_sent.shape


# Still use tanh 

# In[ ]:


def gru_embedding_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):   
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size+1,output_dim=128,input_length=input_shape[1:][0]))
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True,recurrent_dropout=True))
    model.add(TimeDistributed(Dense(1024, activation='tanh')))
    model.add(Dropout(0.6))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(0.001),
                  metrics=['accuracy','sparse_categorical_crossentropy'])
    return model


# In[ ]:


modele = gru_embedding_model(padding_english_sent.shape,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))
modele.summary()


# In[ ]:


history=modele.fit(padding_english_sent,preproc_french_sentences, batch_size=1024, epochs=35,validation_split=0.2,callbacks=[TQDMNotebookCallback()])


# In[ ]:


axes=plot_model_performance('accuracy','loss')


# # Word Embeddings Visualized

# In[ ]:


weights=np.asarray(modele.layers[0].get_weights())
W=weights.reshape(200,128)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Wnorm=sc.fit_transform(W)


# In[ ]:


l=list(english_tokenizer.word_index.keys())


# Embedding Layer Visualized

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(40,30))  
ax.set_yticklabels(english_tokenizer.word_index.keys()) 
ax.set_xticklabels(np.arange(0,128))
sns.heatmap(Wnorm,cmap="Accent_r",linewidths=1,square=True,ax=ax)


# In[ ]:


import matplotlib.pyplot as plt
sns.clustermap(W,cmap='Accent',metric="cosine")


# In[ ]:


from mpl_toolkits.mplot3d import axes3d
from sklearn.manifold import TSNE
import seaborn as sns
tsne=TSNE(n_components=2, perplexity=5,  learning_rate=200.0, n_iter=10000, metric='cosine', verbose=1,random_state=24)
X=tsne.fit_transform(W)
words = list(english_tokenizer.word_index.keys())


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[ ]:


from sklearn.manifold import TSNE
import seaborn as sns
tsne=TSNE(n_components=3, perplexity=5,  learning_rate=200.0, n_iter=10000, metric='cosine', verbose=1,random_state=24)
X=tsne.fit_transform(W)
words = list(english_tokenizer.word_index.keys())


# In[ ]:


from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
x=X[:,0]
y=X[:,1]
z=X[:,2]

ax.scatter(x,y,z,c=z,cmap='hsv')
for x,y,z,i in zip(x,y,z,range(len(words))):
    ax.text(x,y,z,words[i])
#for i, word in enumerate(words):
#    plt.annotate(word, xy=(X[i, 0], X[i, 1]))
#plt.show()


# # Now final layer weights

# In[ ]:


FrW=modele.layers[4].weights[0]
W=FrW.numpy()
Wfr=W.reshape(344,1024)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(80,70))  
sns.heatmap(Wfr,linewidths=1,square=True,ax=axes)
axes.set_yticklabels(list(french_tokenizer.word_index.keys())) 



# In[ ]:


import matplotlib.pyplot as plt
sns.clustermap(Wfr,cmap='Accent',metric="cosine")


# This honestly made no sense.
# - So let us try and visualize them in 3D

# In[ ]:


tsne=TSNE(n_components=3, perplexity=5,  learning_rate=200.0, n_iter=10000, metric='cosine', verbose=1,random_state=24)
Xfr=tsne.fit_transform(Wfr)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')


# Basically trying to interpret the model by visualizing the last layer

# In[ ]:


ax = plt.axes(projection='3d')
x=Xfr[:,0]
y=Xfr[:,1]
z=Xfr[:,2]
wordsfr=list(french_tokenizer.word_index.keys())
ax.scatter(x,y,z,c=z,cmap='hsv')
for x,y,z,i in zip(x,y,z,range(len(wordsfr))):
    ax.text(x,y,z,wordsfr[i])


# # PreTrained Word Embeddings

# Description of Pre-Trained Embeddings
# > The pre-trained Google word2vec model was trained on Google news data (about 100 billion words); it contains 3 million words and phrases and was fit using 300-dimensional word vectors.
# 
# I try to use Word2Vec pretrained embeddings from google in custom GRU model. Then compare these Word2Vec embeddings to the embeddings created by older GRU model

# Thx to KerasDocs for explaining how tio load them .
# Was particulary keen on Gensim having used them b4

# In[ ]:


from gensim.models.keyedvectors import KeyedVectors
filename = '/kaggle/input/googlevec/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)


# In[ ]:


def get_Vector(str):
    if str in model:
             return model[str][:256]
    else:
        return None


# In[ ]:


s=len(english_tokenizer.word_index)+1


# In[ ]:


embedding_matrix = np.zeros((s,256))
for word, i in english_tokenizer.word_index.items():
         embedding_vector = get_Vector(word)
         if embedding_vector is not None:
                 embedding_matrix[i] = embedding_vector
        
            


# In[ ]:


from keras.initializers import Constant


# In[ ]:


def gru_pretrained_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size+1,output_dim=256,input_length=input_shape[1:][0],embeddings_initializer=Constant(embedding_matrix),trainable=False))
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='tanh')))
    model.add(Dropout(0.6))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(0.001),
                  metrics=['accuracy','sparse_categorical_crossentropy'])
    return model


# In[ ]:


modele = gru_pretrained_model(padding_english_sent.shape,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))
modele.summary()


# In[ ]:


history=modele.fit(padding_english_sent,preproc_french_sentences, batch_size=1024, epochs=35,validation_split=0.2,callbacks=[TQDMNotebookCallback()])


# In[ ]:


res=logits_to_text(modele.predict(padding_english_sent[:1])[0], french_tokenizer).replace('<PAD>','').strip()


# In[ ]:


res


# In[ ]:


print(padding_english_sent[:1][0],end="\n\n")
print(preproc_french_sentences[:1][0].reshape(1,21))


# In[ ]:


logits_to_text(preproc_french_sentences[:1][0].reshape(1,21),french_tokenizer)


# In[ ]:


axes=plot_model_performance('accuracy','loss')


# In[ ]:


#weights=np.asarray(modele.layers[0].get_weights())
#W2Vec=weights.reshape(200,128)
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#W2Vecnorm=sc.fit_transform(W2Vec)


# In[ ]:


#import matplotlib.pyplot as plt
#import seaborn as sns
#fig, ax = plt.subplots(figsize=(40,30))  
#sns.heatmap(W2Vecnorm,cmap="Accent_r",linewidths=1,square=True,ax=ax)
#ax.set_yticklabels(list(english_tokenizer.word_index.keys())) 
#ax.set_xticklabels(np.arange(0,128))


# In[ ]:


#from sklearn.metrics.pairwise import cosine_similarity as cosine
#list=[]
#for i in tqdm(range(0,200)):
#    list.append(cosine(Wnorm[i].reshape(1,128),W2Vecnorm[i].reshape(1,128)))


# # PART II
# 
# As we can clearly see that the loss metrics for both the GRU and LSTM RNN's are really high despite an approx accuracy around 82.xx%
# With this aim I try and explore other models which can further reduce the SPRSCATCRENTRPY Loss
# 
# ## BiGRU

# In[ ]:


"""def bidirectional_gru_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):   
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size+1,output_dim=128,input_length=input_shape[1:][0]))
    model.add(Bidirectional(LSTM(256, input_shape=input_shape[1:], return_sequences=True,recurrent_dropout=0.2)))
    model.add(TimeDistributed(Dense(1024, activation='tanh')))
    model.add(Dropout(0.6))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(0.001),
                  metrics=['accuracy','sparse_categorical_crossentropy'])
    return model"""


# In[ ]:


#modelBi=bidirectional_gru_model(padding_english_sent.shape,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))
#modelBi.summary()


# In[ ]:


#history=modelBi.fit(padding_english_sent,preproc_french_sentences, batch_size=1024, epochs=35,validation_split=0.2,callbacks=[TQDMNotebookCallback()])


# In[ ]:


#axes=plot_model_performance('accuracy','loss')


# In[ ]:


#modelBi.save('BILSTMRNN.h5')


# # Part III 
# Encoder-Decoder models .
# Consisting of a rnn acting as encoder and another acting as decoder

# In[ ]:



import tensorflow
tensorflow.random.set_seed(1)

from numpy.random import seed
seed(1)


# In[ ]:



def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    embedding_size = 128
    rnn_cells = 200
    dropout = 0.0
    learning_rate = 1e-3
    from keras.layers import LSTM
    encoder_input_seq = Input(shape=input_shape[1:], name="enc_input")
 
    # Encoder (Return the internal states of the RNN -> 1 hidden state for GRU cells, 2 hidden states for LSTM cells))
    encoder_output, state_t = GRU(units=rnn_cells, 
                                  dropout=dropout,
                                  return_sequences=False,
                                  return_state=True,
                                  name="enc_rnn")(encoder_input_seq)
          #or for LSTM cells: encoder_output, state_h, state_c = LSTM(...)
        
    # Decoder Input   
    decoder_input_seq = RepeatVector(output_sequence_length)(encoder_output)

    # Decoder RNN (Take the encoder returned states as initial states)
    decoder_out = GRU(units=rnn_cells,
                      dropout=dropout,
                      return_sequences=True,
                      return_state=False)(decoder_input_seq, initial_state=state_t)
                                         #or for LSTM cells: (decoder_input_seq, initial_state=[state_h, state_c])
    
    # Decoder output 
    logits = TimeDistributed(Dense(units=french_vocab_size))(decoder_out) 
    
    # Model
    model = Model(encoder_input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
     
    return model    


# In[ ]:


tmp_x = pad(preproc_english_sentences,preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
encdec_rnn_model = encdec_model(input_shape = tmp_x.shape,
                                output_sequence_length =preproc_french_sentences.shape[1],
                                english_vocab_size = english_vocab_size+1,
                                french_vocab_size = french_vocab_size+1)


# In[ ]:


encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)


# In[ ]:




