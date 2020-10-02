#!/usr/bin/env python
# coding: utf-8

# # Stanford Natural Language Inference
# 
# This is a straight forward attempt at the SNLI dataset. NLI is particularly and even Kaggle's free GPU with it's recently expanded 9 hour time limit is insufficient to optimize a model for it. This model uses Elmo embeddings followed by a series of 1D convolutions and pooling to achieve ~67% test accuracy. Using more complicated model structures and expanded time limits, the results can be much better than this but this serves as a good baseline and easier to follow model. Many papers have been written on the SNLI dataset and I find to be a very fulfilling to study after becoming acquainted with this dataset. Enjoy.
# 
# # The Corpus
# 
# *The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE). We aim for it to serve both as a benchmark for evaluating representational systems for text, especially including those induced by representation learning methods, as well as a resource for developing NLP models of any kind.
# 
# Read the rest here: https://nlp.stanford.edu/projects/snli/ 
# 
# In short, the dataset provides one line of text and one hypothesis for the text. The goal of the model is to decide if the hypothesis contradicts, entails or is neutral to the text. So the example text "*A man inspects the uniform of a figure in some East Asian country.*" with a hypothesis of "*The man is sleeping*" is a contradiction because the man cannot inspect if he is asleep.
# 

# # Imports

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.callbacks import Callback
import tensorflow_hub as hub
import tensorflow as tf
import re

from keras import backend as K
import keras.layers as layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, Embedding, Flatten, Activation, SpatialDropout1D
from keras.layers import Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils
from keras.engine import Layer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM, Add, Reshape
from keras.layers import MaxPooling1D, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'

import re
import math
# set seed
np.random.seed(123)


# # Read in data

# In[ ]:


train = pd.read_csv('../input/stanford-natural-language-inference-corpus/snli_1.0_train.csv')
test = pd.read_csv('../input/stanford-natural-language-inference-corpus/snli_1.0_test.csv')
valid = pd.read_csv('../input/stanford-natural-language-inference-corpus/snli_1.0_dev.csv')


# In[ ]:


print("Training on", train.shape[0], "examples")
print("Validating on", test.shape[0], "examples")
print("Testing on", valid.shape[0], "examples")
train[:10]


# # Preprocessing the data
# 
# There are a few NA values to drop in sentence2 and the gold_label has a few "-". The "-" values are when the 5 votes from the turk participants came out tied, usually caused by very confusingly worded rows, so it is best to remove these as well.

# In[ ]:


train.isnull().sum()


# In[ ]:


train.nunique()


# In[ ]:


train = train.dropna(subset = ['sentence2'])
train = train[train["gold_label"] != "-"]
test = test[test["gold_label"] != "-"]
valid = valid[valid["gold_label"] != "-"]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef get_rnn_data(df):\n    x = {\n        \'sentence1\': df["sentence1"],\n        #\n        \'sentence2\': df["sentence2"],\n        }\n    return x\n\nle = LabelEncoder()\n\nX_train = get_rnn_data(train)\nY_train = np_utils.to_categorical(le.fit_transform(train["gold_label"].values)).astype("int64")\n\nX_valid = get_rnn_data(valid)\nY_valid = np_utils.to_categorical(le.fit_transform(valid["gold_label"].values)).astype("int64")\n\nX_test = get_rnn_data(test)\nY_test = np_utils.to_categorical(le.fit_transform(test["gold_label"].values)).astype("int64")')


# # Make the NLI model
# 
# ## Custom Layers

# In[ ]:


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)
    
#     def get_config(self):
#         config = {'output_dim': self.output_dim}
    
class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape
    
#     def get_config(self):
#         config = {'output_dim': self.output_dim}
        
custom_ob={'ElmoEmbeddingLayer': ElmoEmbeddingLayer, 'NonMasking': NonMasking}


# # Build the Model
# 
# This model first processes each sentence separately using Elmo embeddings and sending it through a series of 1D convolutions and Maxpooling layers before concatenating the results together.

# In[ ]:


def get_model():
    inp1 = Input(shape=(1,), dtype="string", name="sentence1")
    inp2 = Input(shape=(1,), dtype="string", name="sentence2")
    
    def rnn_layer(inp, col):
        x = ElmoEmbeddingLayer()(inp)
        x = NonMasking()(x)
        x = Reshape((1, 1024), input_shape=(1024,))(x)
        x = Conv1D(128, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_1")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(64, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_2")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(64, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_3")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(32, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_4")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(32, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_5")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(16, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_6")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(16, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_7")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(8, kernel_size = 2, padding = "same", kernel_initializer = "glorot_uniform", name=col+"_8")(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        return x

    x = concatenate([
                    rnn_layer(inp1,"sen_1"),
                    rnn_layer(inp2,"sen_2"),
                     ])
    x = Flatten()(x)
    x = Dense(8, kernel_initializer='normal', activation='relu', name="final_den_1") (x)
    outp = Dense(3, activation="sigmoid", name="final_output")(x)
    
    model = Model(inputs=[inp1,inp2], outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'],
                 )

    return model

model = get_model()

model.summary()


# # Callbacks 

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)
file_path="checkpoint_SNLI_weights.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

early = EarlyStopping(monitor="val_acc", mode="max", patience=1)

model_callbacks = [checkpoint, early, learning_rate_reduction]


# In[ ]:


# model = load_model("../input/snli-model-and-weights/SNLI_model.h5", custom_objects=custom_ob)
# model.load_weights('../input/snli-model-and-weights/SNLI_weights.hdf5')


# # Train the Model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, Y_train,\n          batch_size=128,\n          epochs=4,\n          verbose=2,\n          validation_data=(X_valid, Y_valid),\n          callbacks = model_callbacks)')


# In[ ]:


model.save_weights("SNLI_weights.hdf5")
model.save("SNLI_model.h5")


# # Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_pred = model.predict(X_test, batch_size=128)')


# In[ ]:


test_acc = (np.argmax(test_pred, axis=1) == np.argmax(Y_test, axis=1)).sum()/Y_test.shape[0] * 100

print("Accuracy on test set is: %"+str(test_acc))


# If you enjoyed this notebook, please like, comment, and check out some of my other notebooks on Kaggle: 
# 
# Making AI Dance Videos: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance
# 
# Image Colorization: https://www.kaggle.com/valkling/image-colorization-using-autoencoders-and-resnet/notebook
# 
# Star Wars Steganography: https://www.kaggle.com/valkling/steganography-hiding-star-wars-scripts-in-images
