#!/usr/bin/env python
# coding: utf-8

# ### What about this kernel?: Understand how to bulild, train and test a CNN for text classification with pretrained word2vec weights.
# * Pre-trained word2vec model: [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
# * Dataset (train+test): [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification)
# * Model architecture: The same as in [the Yoon Kim paper]( https://arxiv.org/pdf/1408.5882.pdf). 
# 

# ### Outline
# 1. Python imports for coding
# 1. Global variables
# 1. Functions for plotting and visualize data
# 1. Preprocessing functions
# 1. Functions to cover the word2vec workaround
# 1. Functions for model management
# 1. Running all

# ### 1. Python imports for coding

# In[ ]:


#General imports to work
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import logging as log # local library for processing logs along the sourcecode

# imports for plot_history_model
import matplotlib.pyplot as plt

#data preprocesing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#imports for word2vec model
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Embedding

#imports for CNN model
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers

#Imports to support keras GPU
from keras import backend as K
import tensorflow as tf
import multiprocessing

#Imports for ploting models
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot


# ### 2. Global variables

# In[ ]:


#several configurations
NUM_WORDS=20000 #max unique words in vocabulary
EMBEDDING_DIM=300 #max neurons for hidden layer in word2vec

#Log format and config
log_format = '%(asctime)s [%(levelname)s] %(message)s'
date_format = '%m-%d-%y %H:%M'
log.basicConfig(level=log.INFO, format=log_format, datefmt=date_format)

#Set your dataset: https://www.kaggle.com/c/spooky-author-identification
training_dataset = '../input/spooky-author-identification/train.csv'
testing_dataset = '../input/spooky-author-identification/test.csv'


# ### 3. Functions for plotting and visualize data

# In[ ]:


#show data info
def get_data_info3D(data,show_cols={1,2}):
    df = pd.DataFrame()
    keys = train_data.keys()
    for i in show_cols:
        if i == len(keys)-1:      
            df.insert(loc=len(df.keys()),column=keys[i]+" (count)",value=[len([data.get(keys[i]).unique()][0])])
            df.insert(loc=len(df.keys()),column=keys[i]+" (list)",value=[([data.get(keys[i]).unique()][0])])
        else:
           df.insert(loc=len(df.keys()),column=keys[i],value=[data.get(keys[i]).count()])  
    return df


# In[ ]:


#https://keras.io/visualization/
#Plot history model results once the fit process has been concluded
def plot_history_model_of_training_process(history):
    #Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# ### 4. Preprocessing functions

# In[ ]:


def preprocesing_data(NUM_WORDS,train_data):
    #Ok, seems clean, lets make categories of authors
    authors=train_data.author.unique()
    dic={}
    for i,author in enumerate(authors):
        dic[author]=i
    labels=train_data.author.apply(lambda x:dic[x])
    
    #NumPy-style indexing doesn't work on a Pandas DataFrame; use
    #Lets divide our training data in train and validation
    val_data=train_data.sample(frac=0.2,random_state=200)
    train_data=train_data.drop(val_data.index)

    #Tokenizing training data text with keras text preprocessing functions
    texts=train_data.text
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid=tokenizer.texts_to_sequences(val_data.text)
    word_index = tokenizer.word_index
    log.info('Found {0} unique tokens.'.format(len(word_index)))

    X_train = pad_sequences(sequences_train)
    X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
    y_train = to_categorical(np.asarray(labels[train_data.index]))
    y_val = to_categorical(np.asarray(labels[val_data.index]))
    log.info('Shape of train: {0} and validation tensor: {1} '.format(X_train.shape,X_val.shape))
    log.info('Shape of label train: {0} and validation tensor: {1}'.format(y_train.shape,y_val.shape))
    return tokenizer,dic,word_index,X_train,X_val,y_train,y_val


# ### 5. Functions to cover the word2vec workaround
# Now its time for word embedding. To perform this, we are going to use a pretrained word2vec model, just choice your best way: 
# 1.  Just pick on "+ Add Data" button and look for "Google news vector". it will be attached to your current kernel, dont worry about not enough space. 
# 2.  Download the dataset directly from the google storage: [link_to_google_storage](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). Then just upload it to your kaggle environment (use again the "+ Add Data" button on the right side.

# In[ ]:


def load_pretrained_word2vec(word_index,NUM_WORDS):
    word_vectors = KeyedVectors.load_word2vec_format('../input/gnewsvector/GoogleNews-vectors-negative300.bin', binary=True)
    vocabulary_size=min(len(word_index)+1,NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i>=NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
    del(word_vectors)
    embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)
    return embedding_layer
#TODO: Code fine tunning
#TODO: Code from scratch
#Case in which we dont have pretraining word vectors. 
#from keras.layers import Embedding
#EMBEDDING_DIM=300
#vocabulary_size=min(len(word_index)+1,NUM_WORDS)

#embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM)


# ### 6. Functions for model management

# In[ ]:


#Adding support to select between CPU or GPU on Keras backend
#Modes: use_gpu=0 (just use CPU), use_gpu=1 (use gpu when its prossible)
def set_hardware_backend(use_gpu=0):
    #setting max CPU cores
    max_cpu_cores = multiprocessing.cpu_count()
    #configuring tensorflow protocol to choice between CPU|GPU, burning the hardware!!!
    config = tf.ConfigProto(intra_op_parallelism_threads=max_cpu_cores,
                            inter_op_parallelism_threads=max_cpu_cores, 
                            allow_soft_placement=True,
                            device_count = {'GPU': use_gpu, 'CPU': max_cpu_cores})
    #Only for shared GPU's: allocates the GPU memory dynamically instead of all at time, 
    config.gpu_options.allow_growth=True 
    sess = tf.Session(graph=tf.get_default_graph(),config=config) 
    K.set_session(sess)


# In[ ]:


def create_model(X_train,EMBEDDING_DIM,embedding_layer):
    #Hyperparameters
    filter_sizes = [3,4,5] #defined convs regions
    num_filters = 100 #num_filters per conv region
    drop = 0.5

    sequence_length = X_train.shape[1]
    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3*num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=3, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

    # this creates a model that includes
    model = Model(inputs, output)
    return model


# In[ ]:


#We have to implement SGD-Adadelta-udpate-rule (Zeiler, 2012)
#Right now we are using ADAM instead of SGD
def training_model(model,X_train,y_train,X_val,y_val):
    #Learning rate
    adam = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    callbacks = [EarlyStopping(monitor='val_loss')]
    # starts training
    history = model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)
    return history


# In[ ]:


def testing_model(test_data,X_train,tokenizer):
    #tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True) #extract this function from here
    #there is a problem in this tokenizer because it should be the same used in training dataset, arrange it tomorrow!!! 
    sequences_test=tokenizer.texts_to_sequences(test_data.text) #the same here, this is preprocessing
    X_test = pad_sequences(sequences_test,maxlen=X_train.shape[1]) #Equalizes X_test texts length to X_train
    y_pred=model.predict(X_test)
    return y_pred


# ### 7. Running all
# 

# In[ ]:


log.info("Loading datasets...")
train_data=pd.read_csv(training_dataset)
test_data=pd.read_csv(testing_dataset)

#log.info("Showing datasets info...")
#print(get_data_info3D(train_data))
#print(get_data_info3D(test_data,{1}))


# In[ ]:


log.info("Preprocesing input...")
tokenizer,dic,word_index,X_train,X_val,y_train,y_val = preprocesing_data(NUM_WORDS,train_data)
embedding_layer = load_pretrained_word2vec(word_index,NUM_WORDS)


# In[ ]:


log.info("Creating model...")
set_hardware_backend(use_gpu=1) #we set the backend to GPU to descrease the training time
model = create_model(X_train,EMBEDDING_DIM,embedding_layer)
model.summary() #showing layers, parameteres and connections
SVG(model_to_dot(model).create(prog='dot', format='svg')) #plotting the model graph


# In[ ]:


log.info("training model... it will take a while.")
history = training_model(model,X_train,y_train,X_val,y_val)
plot_history_model_of_training_process(history)


# In[ ]:


log.info("testing model...")
y_pred = testing_model(test_data,X_train,tokenizer)


# In[ ]:


#TODO: Good idea to have the confusion matrix at the end: https://scikit-learn.org/stable/ and pyimagesearch (some example)
to_submit=pd.DataFrame(index=test_data.id,data={'EAP':y_pred[:,dic['EAP']],
                                                'HPL':y_pred[:,dic['HPL']],
                                                'MWS':y_pred[:,dic['MWS']]})


# In[ ]:


to_submit.to_csv('../input/submit.csv')
to_submit.head(5) #showing results over all testing dataset. Most higher values in each category means caterogized as it is. 
                  #TODO: get class colum for testing dataset in order to create the confusion matrix plus metrics

