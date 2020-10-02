#!/usr/bin/env python
# coding: utf-8

# Import 

# In[ ]:


from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, SpatialDropout1D, Conv1D,MaxPool1D,BatchNormalization, GRU, SimpleRNN, Dropout,Flatten, concatenate,Bidirectional, GlobalMaxPool1D
from tensorflow.keras import utils 
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
import matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import random
import time
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import brown
from textblob import TextBlob
from tensorflow.keras import regularizers


# In[ ]:


import nltk
nltk.download('wordnet')


# In[ ]:


stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


train = pd.read_csv('/content/drive/My Drive/kaggle/train (1).csv')


# In[ ]:


get_ipython().system("unzip '/content/drive/My Drive/glove.6B.100d.zip'")


# In[ ]:


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


# In[ ]:


train['textNltk'] = train['text'].apply(lambda x: clean_text(x))


# In[ ]:


maxWordsCount = 20000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', oov_token='unknown', char_level=False)
tokenizer.fit_on_texts(train['textNltk'])
word_index = tokenizer.word_index
vocab_size = max_features = len(word_index)


# In[ ]:


trainWordIndexes = tokenizer.texts_to_sequences(train['textNltk']) 


# In[ ]:


path_file = 'glove.6B.100d.txt'
embedding_dict_100d={}
with open(path_file,'r', encoding='utf8') as f:
    for line in f:
        values=line.split(' ')
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict_100d[word]=vectors
f.close()

print('The size of the embedding index is : %d'%(len(embedding_dict_100d)))


# In[ ]:


embedding_dim  = 100

embedding_matrix = np.zeros((max_features+1, embedding_dim))

for word, i in word_index.items():
  if i < max_features:
    
    try:
      embedding_matrix[i] =embedding_dict_100d[word]
    except KeyError:  
      embedding_matrix[i] = np.zeros((1, embedding_dim))
embedding_matrix.shape


# In[ ]:


maxLen = 25
XEmb = pad_sequences(trainWordIndexes, maxlen=maxLen,padding='post')
y = np.asarray(train['target'])


# In[ ]:


xTrain, xVal, yTrain, yVal = train_test_split(XEmb, y, test_size=0.20)


# In[ ]:


def plots_visual(result_model):
  fig, axes = plt.subplots(1, 2, figsize=(13,5))
  axes[0].plot(result_model.history['accuracy'])
  axes[0].plot(result_model.history['val_accuracy'])
  axes[0].legend(['accuracy', 'val_accuracy'])
  axes[0].set_title('Accuracy')
  axes[0].set_xlabel('Epochs')
  axes[1].plot(result_model.history['loss'])
  axes[1].plot(result_model.history['val_loss'])
  axes[1].legend(['loss', 'val_loss'])
  axes[1].set_title('Loss')
  axes[1].set_xlabel('Epochs')


# In[ ]:


from keras.initializers import Constant


# In[ ]:


#Base Model EMB + LSTM


# In[ ]:


def models():
    model = Sequential()
    model.add(Embedding(max_features+1, embedding_dim, mask_zero=True,embeddings_initializer=Constant(embedding_matrix)))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences = True, dropout=0.3, recurrent_dropout=0.5)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    model.summary()
    return model

model = models()


# In[ ]:


history = model.fit(xTrain,yTrain,epochs=150,batch_size=32, validation_data=(xVal, yVal), callbacks=[callbaks])


# In[ ]:


Epoch 1/150
191/191 [==============================] - ETA: 0s - loss: 0.7184 - accuracy: 0.5103
Epoch 00001: val_accuracy improved from -inf to 0.65266, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 207ms/step - loss: 0.7184 - accuracy: 0.5103 - val_loss: 0.6653 - val_accuracy: 0.6527 - lr: 1.0000e-04
Epoch 2/150
191/191 [==============================] - ETA: 0s - loss: 0.6821 - accuracy: 0.5649
Epoch 00002: val_accuracy improved from 0.65266 to 0.70519, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 209ms/step - loss: 0.6821 - accuracy: 0.5649 - val_loss: 0.6378 - val_accuracy: 0.7052 - lr: 1.0000e-04
Epoch 3/150
191/191 [==============================] - ETA: 0s - loss: 0.6642 - accuracy: 0.6007
Epoch 00003: val_accuracy improved from 0.70519 to 0.73145, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 207ms/step - loss: 0.6642 - accuracy: 0.6007 - val_loss: 0.6155 - val_accuracy: 0.7315 - lr: 1.0000e-04
Epoch 4/150
191/191 [==============================] - ETA: 0s - loss: 0.6531 - accuracy: 0.6246
Epoch 00004: val_accuracy improved from 0.73145 to 0.75837, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 208ms/step - loss: 0.6531 - accuracy: 0.6246 - val_loss: 0.5940 - val_accuracy: 0.7584 - lr: 1.0000e-04
Epoch 5/150
191/191 [==============================] - ETA: 0s - loss: 0.6335 - accuracy: 0.6445
Epoch 00005: val_accuracy improved from 0.75837 to 0.77741, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 207ms/step - loss: 0.6335 - accuracy: 0.6445 - val_loss: 0.5735 - val_accuracy: 0.7774 - lr: 1.0000e-04
Epoch 6/150
191/191 [==============================] - ETA: 0s - loss: 0.6193 - accuracy: 0.6662
Epoch 00006: val_accuracy improved from 0.77741 to 0.78070, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 207ms/step - loss: 0.6193 - accuracy: 0.6662 - val_loss: 0.5541 - val_accuracy: 0.7807 - lr: 1.0000e-04
Epoch 7/150
191/191 [==============================] - ETA: 0s - loss: 0.5992 - accuracy: 0.6897
Epoch 00007: val_accuracy improved from 0.78070 to 0.78726, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 208ms/step - loss: 0.5992 - accuracy: 0.6897 - val_loss: 0.5351 - val_accuracy: 0.7873 - lr: 1.0000e-04
Epoch 8/150
191/191 [==============================] - ETA: 0s - loss: 0.5909 - accuracy: 0.6969
Epoch 00008: val_accuracy improved from 0.78726 to 0.78989, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 207ms/step - loss: 0.5909 - accuracy: 0.6969 - val_loss: 0.5188 - val_accuracy: 0.7899 - lr: 1.0000e-04
Epoch 9/150
191/191 [==============================] - ETA: 0s - loss: 0.5778 - accuracy: 0.7118
Epoch 00009: val_accuracy did not improve from 0.78989
191/191 [==============================] - 38s 198ms/step - loss: 0.5778 - accuracy: 0.7118 - val_loss: 0.5046 - val_accuracy: 0.7866 - lr: 1.0000e-04
Epoch 10/150
191/191 [==============================] - ETA: 0s - loss: 0.5689 - accuracy: 0.7154
Epoch 00010: val_accuracy did not improve from 0.78989
191/191 [==============================] - 38s 198ms/step - loss: 0.5689 - accuracy: 0.7154 - val_loss: 0.4937 - val_accuracy: 0.7866 - lr: 1.0000e-04
Epoch 11/150
191/191 [==============================] - ETA: 0s - loss: 0.5646 - accuracy: 0.7140
Epoch 00011: val_accuracy did not improve from 0.78989
191/191 [==============================] - 37s 196ms/step - loss: 0.5646 - accuracy: 0.7140 - val_loss: 0.4857 - val_accuracy: 0.7886 - lr: 1.0000e-04
Epoch 12/150
191/191 [==============================] - ETA: 0s - loss: 0.5670 - accuracy: 0.7182
Epoch 00012: val_accuracy did not improve from 0.78989
191/191 [==============================] - 37s 194ms/step - loss: 0.5670 - accuracy: 0.7182 - val_loss: 0.4795 - val_accuracy: 0.7886 - lr: 1.0000e-04
Epoch 13/150
191/191 [==============================] - ETA: 0s - loss: 0.5529 - accuracy: 0.7304
Epoch 00013: val_accuracy improved from 0.78989 to 0.79120, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 206ms/step - loss: 0.5529 - accuracy: 0.7304 - val_loss: 0.4743 - val_accuracy: 0.7912 - lr: 1.0000e-04
Epoch 14/150
191/191 [==============================] - ETA: 0s - loss: 0.5421 - accuracy: 0.7376
Epoch 00014: val_accuracy improved from 0.79120 to 0.79317, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 207ms/step - loss: 0.5421 - accuracy: 0.7376 - val_loss: 0.4702 - val_accuracy: 0.7932 - lr: 1.0000e-04
Epoch 15/150
191/191 [==============================] - ETA: 0s - loss: 0.5363 - accuracy: 0.7473
Epoch 00015: val_accuracy improved from 0.79317 to 0.79383, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 209ms/step - loss: 0.5363 - accuracy: 0.7473 - val_loss: 0.4671 - val_accuracy: 0.7938 - lr: 1.0000e-04
Epoch 16/150
191/191 [==============================] - ETA: 0s - loss: 0.5279 - accuracy: 0.7435
Epoch 00016: val_accuracy did not improve from 0.79383
191/191 [==============================] - 38s 198ms/step - loss: 0.5279 - accuracy: 0.7435 - val_loss: 0.4653 - val_accuracy: 0.7938 - lr: 1.0000e-04
Epoch 17/150
191/191 [==============================] - ETA: 0s - loss: 0.5328 - accuracy: 0.7476
Epoch 00017: val_accuracy improved from 0.79383 to 0.79514, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 204ms/step - loss: 0.5328 - accuracy: 0.7476 - val_loss: 0.4627 - val_accuracy: 0.7951 - lr: 1.0000e-04
Epoch 18/150
191/191 [==============================] - ETA: 0s - loss: 0.5335 - accuracy: 0.7450
Epoch 00018: val_accuracy did not improve from 0.79514
191/191 [==============================] - 38s 197ms/step - loss: 0.5335 - accuracy: 0.7450 - val_loss: 0.4592 - val_accuracy: 0.7951 - lr: 1.0000e-04
Epoch 19/150
191/191 [==============================] - ETA: 0s - loss: 0.5252 - accuracy: 0.7493
Epoch 00019: val_accuracy improved from 0.79514 to 0.79580, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 204ms/step - loss: 0.5252 - accuracy: 0.7493 - val_loss: 0.4574 - val_accuracy: 0.7958 - lr: 1.0000e-04
Epoch 20/150
191/191 [==============================] - ETA: 0s - loss: 0.5251 - accuracy: 0.7502
Epoch 00020: val_accuracy improved from 0.79580 to 0.79711, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 208ms/step - loss: 0.5251 - accuracy: 0.7502 - val_loss: 0.4546 - val_accuracy: 0.7971 - lr: 1.0000e-04
Epoch 21/150
191/191 [==============================] - ETA: 0s - loss: 0.5189 - accuracy: 0.7502
Epoch 00021: val_accuracy improved from 0.79711 to 0.79777, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.5189 - accuracy: 0.7502 - val_loss: 0.4526 - val_accuracy: 0.7978 - lr: 1.0000e-04
Epoch 22/150
191/191 [==============================] - ETA: 0s - loss: 0.5201 - accuracy: 0.7540
Epoch 00022: val_accuracy improved from 0.79777 to 0.80039, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 207ms/step - loss: 0.5201 - accuracy: 0.7540 - val_loss: 0.4521 - val_accuracy: 0.8004 - lr: 1.0000e-04
Epoch 23/150
191/191 [==============================] - ETA: 0s - loss: 0.5146 - accuracy: 0.7601
Epoch 00023: val_accuracy improved from 0.80039 to 0.80105, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 208ms/step - loss: 0.5146 - accuracy: 0.7601 - val_loss: 0.4518 - val_accuracy: 0.8011 - lr: 1.0000e-04
Epoch 24/150
191/191 [==============================] - ETA: 0s - loss: 0.5085 - accuracy: 0.7609
Epoch 00024: val_accuracy did not improve from 0.80105
191/191 [==============================] - 38s 197ms/step - loss: 0.5085 - accuracy: 0.7609 - val_loss: 0.4477 - val_accuracy: 0.8004 - lr: 1.0000e-04
Epoch 25/150
191/191 [==============================] - ETA: 0s - loss: 0.5016 - accuracy: 0.7703
Epoch 00025: val_accuracy improved from 0.80105 to 0.80171, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.5016 - accuracy: 0.7703 - val_loss: 0.4489 - val_accuracy: 0.8017 - lr: 1.0000e-04
Epoch 26/150
191/191 [==============================] - ETA: 0s - loss: 0.5039 - accuracy: 0.7685
Epoch 00026: val_accuracy did not improve from 0.80171
191/191 [==============================] - 37s 196ms/step - loss: 0.5039 - accuracy: 0.7685 - val_loss: 0.4470 - val_accuracy: 0.8011 - lr: 1.0000e-04
Epoch 27/150
191/191 [==============================] - ETA: 0s - loss: 0.5006 - accuracy: 0.7668
Epoch 00027: val_accuracy improved from 0.80171 to 0.80499, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 206ms/step - loss: 0.5006 - accuracy: 0.7668 - val_loss: 0.4444 - val_accuracy: 0.8050 - lr: 1.0000e-04
Epoch 28/150
191/191 [==============================] - ETA: 0s - loss: 0.4979 - accuracy: 0.7688
Epoch 00028: val_accuracy improved from 0.80499 to 0.80565, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 204ms/step - loss: 0.4979 - accuracy: 0.7688 - val_loss: 0.4456 - val_accuracy: 0.8056 - lr: 1.0000e-04
Epoch 29/150
191/191 [==============================] - ETA: 0s - loss: 0.4996 - accuracy: 0.7734
Epoch 00029: val_accuracy improved from 0.80565 to 0.80762, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 207ms/step - loss: 0.4996 - accuracy: 0.7734 - val_loss: 0.4444 - val_accuracy: 0.8076 - lr: 1.0000e-04
Epoch 30/150
191/191 [==============================] - ETA: 0s - loss: 0.4948 - accuracy: 0.7729
Epoch 00030: val_accuracy did not improve from 0.80762
191/191 [==============================] - 38s 197ms/step - loss: 0.4948 - accuracy: 0.7729 - val_loss: 0.4429 - val_accuracy: 0.8076 - lr: 1.0000e-04
Epoch 31/150
191/191 [==============================] - ETA: 0s - loss: 0.4906 - accuracy: 0.7813
Epoch 00031: val_accuracy did not improve from 0.80762
191/191 [==============================] - 38s 197ms/step - loss: 0.4906 - accuracy: 0.7813 - val_loss: 0.4424 - val_accuracy: 0.8076 - lr: 1.0000e-04
Epoch 32/150
191/191 [==============================] - ETA: 0s - loss: 0.4872 - accuracy: 0.7757
Epoch 00032: val_accuracy improved from 0.80762 to 0.81090, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 207ms/step - loss: 0.4872 - accuracy: 0.7757 - val_loss: 0.4394 - val_accuracy: 0.8109 - lr: 1.0000e-04
Epoch 33/150
191/191 [==============================] - ETA: 0s - loss: 0.4881 - accuracy: 0.7775
Epoch 00033: val_accuracy did not improve from 0.81090
191/191 [==============================] - 38s 198ms/step - loss: 0.4881 - accuracy: 0.7775 - val_loss: 0.4377 - val_accuracy: 0.8109 - lr: 1.0000e-04
Epoch 34/150
191/191 [==============================] - ETA: 0s - loss: 0.4836 - accuracy: 0.7757
Epoch 00034: val_accuracy improved from 0.81090 to 0.81221, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4836 - accuracy: 0.7757 - val_loss: 0.4387 - val_accuracy: 0.8122 - lr: 1.0000e-04
Epoch 35/150
191/191 [==============================] - ETA: 0s - loss: 0.4868 - accuracy: 0.7750
Epoch 00035: val_accuracy improved from 0.81221 to 0.81418, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 204ms/step - loss: 0.4868 - accuracy: 0.7750 - val_loss: 0.4373 - val_accuracy: 0.8142 - lr: 1.0000e-04
Epoch 36/150
191/191 [==============================] - ETA: 0s - loss: 0.4796 - accuracy: 0.7757
Epoch 00036: val_accuracy improved from 0.81418 to 0.81550, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4796 - accuracy: 0.7757 - val_loss: 0.4379 - val_accuracy: 0.8155 - lr: 1.0000e-04
Epoch 37/150
191/191 [==============================] - ETA: 0s - loss: 0.4768 - accuracy: 0.7872
Epoch 00037: val_accuracy did not improve from 0.81550
191/191 [==============================] - 38s 198ms/step - loss: 0.4768 - accuracy: 0.7872 - val_loss: 0.4365 - val_accuracy: 0.8148 - lr: 1.0000e-04
Epoch 38/150
191/191 [==============================] - ETA: 0s - loss: 0.4797 - accuracy: 0.7824
Epoch 00038: val_accuracy did not improve from 0.81550
191/191 [==============================] - 37s 196ms/step - loss: 0.4797 - accuracy: 0.7824 - val_loss: 0.4356 - val_accuracy: 0.8155 - lr: 1.0000e-04
Epoch 39/150
191/191 [==============================] - ETA: 0s - loss: 0.4735 - accuracy: 0.7882
Epoch 00039: val_accuracy did not improve from 0.81550
191/191 [==============================] - 38s 199ms/step - loss: 0.4735 - accuracy: 0.7882 - val_loss: 0.4345 - val_accuracy: 0.8155 - lr: 1.0000e-04
Epoch 40/150
191/191 [==============================] - ETA: 0s - loss: 0.4742 - accuracy: 0.7870
Epoch 00040: val_accuracy improved from 0.81550 to 0.81878, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4742 - accuracy: 0.7870 - val_loss: 0.4330 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 41/150
191/191 [==============================] - ETA: 0s - loss: 0.4689 - accuracy: 0.7903
Epoch 00041: val_accuracy did not improve from 0.81878
191/191 [==============================] - 38s 198ms/step - loss: 0.4689 - accuracy: 0.7903 - val_loss: 0.4327 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 42/150
191/191 [==============================] - ETA: 0s - loss: 0.4721 - accuracy: 0.7877
Epoch 00042: val_accuracy improved from 0.81878 to 0.81944, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4721 - accuracy: 0.7877 - val_loss: 0.4330 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 43/150
191/191 [==============================] - ETA: 0s - loss: 0.4680 - accuracy: 0.7877
Epoch 00043: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4680 - accuracy: 0.7877 - val_loss: 0.4326 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 44/150
191/191 [==============================] - ETA: 0s - loss: 0.4659 - accuracy: 0.7959
Epoch 00044: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4659 - accuracy: 0.7959 - val_loss: 0.4326 - val_accuracy: 0.8168 - lr: 1.0000e-04
Epoch 45/150
191/191 [==============================] - ETA: 0s - loss: 0.4638 - accuracy: 0.7906
Epoch 00045: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4638 - accuracy: 0.7906 - val_loss: 0.4303 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 46/150
191/191 [==============================] - ETA: 0s - loss: 0.4613 - accuracy: 0.7938
Epoch 00046: val_accuracy did not improve from 0.81944
191/191 [==============================] - 37s 196ms/step - loss: 0.4613 - accuracy: 0.7938 - val_loss: 0.4323 - val_accuracy: 0.8162 - lr: 1.0000e-04
Epoch 47/150
191/191 [==============================] - ETA: 0s - loss: 0.4597 - accuracy: 0.7877
Epoch 00047: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 200ms/step - loss: 0.4597 - accuracy: 0.7877 - val_loss: 0.4323 - val_accuracy: 0.8148 - lr: 1.0000e-04
Epoch 48/150
191/191 [==============================] - ETA: 0s - loss: 0.4573 - accuracy: 0.7985
Epoch 00048: val_accuracy did not improve from 0.81944
191/191 [==============================] - 37s 195ms/step - loss: 0.4573 - accuracy: 0.7985 - val_loss: 0.4308 - val_accuracy: 0.8162 - lr: 1.0000e-04
Epoch 49/150
191/191 [==============================] - ETA: 0s - loss: 0.4531 - accuracy: 0.7977
Epoch 00049: val_accuracy did not improve from 0.81944
191/191 [==============================] - 37s 196ms/step - loss: 0.4531 - accuracy: 0.7977 - val_loss: 0.4300 - val_accuracy: 0.8181 - lr: 1.0000e-04
Epoch 50/150
191/191 [==============================] - ETA: 0s - loss: 0.4514 - accuracy: 0.8002
Epoch 00050: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4514 - accuracy: 0.8002 - val_loss: 0.4287 - val_accuracy: 0.8155 - lr: 1.0000e-04
Epoch 51/150
191/191 [==============================] - ETA: 0s - loss: 0.4516 - accuracy: 0.8007
Epoch 00051: val_accuracy did not improve from 0.81944
191/191 [==============================] - 37s 195ms/step - loss: 0.4516 - accuracy: 0.8007 - val_loss: 0.4274 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 52/150
191/191 [==============================] - ETA: 0s - loss: 0.4490 - accuracy: 0.8011
Epoch 00052: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4490 - accuracy: 0.8011 - val_loss: 0.4285 - val_accuracy: 0.8175 - lr: 1.0000e-04
Epoch 53/150
191/191 [==============================] - ETA: 0s - loss: 0.4461 - accuracy: 0.8033
Epoch 00053: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 198ms/step - loss: 0.4461 - accuracy: 0.8033 - val_loss: 0.4274 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 54/150
191/191 [==============================] - ETA: 0s - loss: 0.4450 - accuracy: 0.8046
Epoch 00054: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 196ms/step - loss: 0.4450 - accuracy: 0.8046 - val_loss: 0.4270 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 55/150
191/191 [==============================] - ETA: 0s - loss: 0.4497 - accuracy: 0.8023
Epoch 00055: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 200ms/step - loss: 0.4497 - accuracy: 0.8023 - val_loss: 0.4274 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 56/150
191/191 [==============================] - ETA: 0s - loss: 0.4380 - accuracy: 0.8021
Epoch 00056: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 200ms/step - loss: 0.4380 - accuracy: 0.8021 - val_loss: 0.4277 - val_accuracy: 0.8181 - lr: 1.0000e-04
Epoch 57/150
191/191 [==============================] - ETA: 0s - loss: 0.4461 - accuracy: 0.8011
Epoch 00057: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 199ms/step - loss: 0.4461 - accuracy: 0.8011 - val_loss: 0.4278 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 58/150
191/191 [==============================] - ETA: 0s - loss: 0.4370 - accuracy: 0.8113
Epoch 00058: val_accuracy did not improve from 0.81944
191/191 [==============================] - 38s 197ms/step - loss: 0.4370 - accuracy: 0.8113 - val_loss: 0.4264 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 59/150
191/191 [==============================] - ETA: 0s - loss: 0.4432 - accuracy: 0.8039
Epoch 00059: val_accuracy improved from 0.81944 to 0.82009, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4432 - accuracy: 0.8039 - val_loss: 0.4271 - val_accuracy: 0.8201 - lr: 1.0000e-04
Epoch 60/150
191/191 [==============================] - ETA: 0s - loss: 0.4404 - accuracy: 0.8107
Epoch 00060: val_accuracy did not improve from 0.82009
191/191 [==============================] - 38s 198ms/step - loss: 0.4404 - accuracy: 0.8107 - val_loss: 0.4271 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 61/150
191/191 [==============================] - ETA: 0s - loss: 0.4321 - accuracy: 0.8123
Epoch 00061: val_accuracy did not improve from 0.82009
191/191 [==============================] - 38s 198ms/step - loss: 0.4321 - accuracy: 0.8123 - val_loss: 0.4267 - val_accuracy: 0.8181 - lr: 1.0000e-04
Epoch 62/150
191/191 [==============================] - ETA: 0s - loss: 0.4351 - accuracy: 0.8077
Epoch 00062: val_accuracy improved from 0.82009 to 0.82075, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 39s 205ms/step - loss: 0.4351 - accuracy: 0.8077 - val_loss: 0.4244 - val_accuracy: 0.8207 - lr: 1.0000e-04
Epoch 63/150
191/191 [==============================] - ETA: 0s - loss: 0.4333 - accuracy: 0.8074
Epoch 00063: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 201ms/step - loss: 0.4333 - accuracy: 0.8074 - val_loss: 0.4256 - val_accuracy: 0.8181 - lr: 1.0000e-04
Epoch 64/150
191/191 [==============================] - ETA: 0s - loss: 0.4278 - accuracy: 0.8154
Epoch 00064: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 197ms/step - loss: 0.4278 - accuracy: 0.8154 - val_loss: 0.4261 - val_accuracy: 0.8194 - lr: 1.0000e-04
Epoch 65/150
191/191 [==============================] - ETA: 0s - loss: 0.4264 - accuracy: 0.8158
Epoch 00065: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 199ms/step - loss: 0.4264 - accuracy: 0.8158 - val_loss: 0.4332 - val_accuracy: 0.8142 - lr: 1.0000e-04
Epoch 66/150
191/191 [==============================] - ETA: 0s - loss: 0.4294 - accuracy: 0.8144
Epoch 00066: val_accuracy did not improve from 0.82075

Epoch 00066: ReduceLROnPlateau reducing learning rate to 7.999999797903001e-05.
191/191 [==============================] - 38s 198ms/step - loss: 0.4294 - accuracy: 0.8144 - val_loss: 0.4290 - val_accuracy: 0.8168 - lr: 1.0000e-04
Epoch 67/150
191/191 [==============================] - ETA: 0s - loss: 0.4291 - accuracy: 0.8153
Epoch 00067: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 198ms/step - loss: 0.4291 - accuracy: 0.8153 - val_loss: 0.4268 - val_accuracy: 0.8194 - lr: 8.0000e-05
Epoch 68/150
191/191 [==============================] - ETA: 0s - loss: 0.4210 - accuracy: 0.8179
Epoch 00068: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 198ms/step - loss: 0.4210 - accuracy: 0.8179 - val_loss: 0.4263 - val_accuracy: 0.8194 - lr: 8.0000e-05
Epoch 69/150
191/191 [==============================] - ETA: 0s - loss: 0.4306 - accuracy: 0.8135
Epoch 00069: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 199ms/step - loss: 0.4306 - accuracy: 0.8135 - val_loss: 0.4240 - val_accuracy: 0.8201 - lr: 8.0000e-05
Epoch 70/150
191/191 [==============================] - ETA: 0s - loss: 0.4247 - accuracy: 0.8167
Epoch 00070: val_accuracy did not improve from 0.82075
191/191 [==============================] - 37s 195ms/step - loss: 0.4247 - accuracy: 0.8167 - val_loss: 0.4243 - val_accuracy: 0.8207 - lr: 8.0000e-05
Epoch 71/150
191/191 [==============================] - ETA: 0s - loss: 0.4146 - accuracy: 0.8232
Epoch 00071: val_accuracy did not improve from 0.82075
191/191 [==============================] - 39s 203ms/step - loss: 0.4146 - accuracy: 0.8232 - val_loss: 0.4268 - val_accuracy: 0.8181 - lr: 8.0000e-05
Epoch 72/150
191/191 [==============================] - ETA: 0s - loss: 0.4218 - accuracy: 0.8236
Epoch 00072: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 200ms/step - loss: 0.4218 - accuracy: 0.8236 - val_loss: 0.4263 - val_accuracy: 0.8194 - lr: 8.0000e-05
Epoch 73/150
191/191 [==============================] - ETA: 0s - loss: 0.4098 - accuracy: 0.8250
Epoch 00073: val_accuracy did not improve from 0.82075

Epoch 00073: ReduceLROnPlateau reducing learning rate to 6.399999838322402e-05.
191/191 [==============================] - 38s 198ms/step - loss: 0.4098 - accuracy: 0.8250 - val_loss: 0.4270 - val_accuracy: 0.8188 - lr: 8.0000e-05
Epoch 74/150
191/191 [==============================] - ETA: 0s - loss: 0.4261 - accuracy: 0.8136
Epoch 00074: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 200ms/step - loss: 0.4261 - accuracy: 0.8136 - val_loss: 0.4266 - val_accuracy: 0.8201 - lr: 6.4000e-05
Epoch 75/150
191/191 [==============================] - ETA: 0s - loss: 0.4101 - accuracy: 0.8227
Epoch 00075: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 199ms/step - loss: 0.4101 - accuracy: 0.8227 - val_loss: 0.4269 - val_accuracy: 0.8201 - lr: 6.4000e-05
Epoch 76/150
191/191 [==============================] - ETA: 0s - loss: 0.4124 - accuracy: 0.8200
Epoch 00076: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 199ms/step - loss: 0.4124 - accuracy: 0.8200 - val_loss: 0.4284 - val_accuracy: 0.8201 - lr: 6.4000e-05
Epoch 77/150
191/191 [==============================] - ETA: 0s - loss: 0.4154 - accuracy: 0.8222
Epoch 00077: val_accuracy did not improve from 0.82075

Epoch 00077: ReduceLROnPlateau reducing learning rate to 5.119999987073243e-05.
191/191 [==============================] - 38s 201ms/step - loss: 0.4154 - accuracy: 0.8222 - val_loss: 0.4273 - val_accuracy: 0.8207 - lr: 6.4000e-05
Epoch 78/150
191/191 [==============================] - ETA: 0s - loss: 0.4113 - accuracy: 0.8261
Epoch 00078: val_accuracy did not improve from 0.82075
191/191 [==============================] - 38s 197ms/step - loss: 0.4113 - accuracy: 0.8261 - val_loss: 0.4259 - val_accuracy: 0.8207 - lr: 5.1200e-05
Epoch 79/150
191/191 [==============================] - ETA: 0s - loss: 0.4110 - accuracy: 0.8245
Epoch 00079: val_accuracy improved from 0.82075 to 0.82141, saving model to /content/drive/My Drive/kaggle/best_model.h5
191/191 [==============================] - 40s 208ms/step - loss: 0.4110 - accuracy: 0.8245 - val_loss: 0.4259 - val_accuracy: 0.8214 - lr: 5.1200e-05
Epoch 80/150
191/191 [==============================] - ETA: 0s - loss: 0.4179 - accuracy: 0.8184
Epoch 00080: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.4179 - accuracy: 0.8184 - val_loss: 0.4273 - val_accuracy: 0.8188 - lr: 5.1200e-05
Epoch 81/150
191/191 [==============================] - ETA: 0s - loss: 0.4106 - accuracy: 0.8223
Epoch 00081: val_accuracy did not improve from 0.82141

Epoch 00081: ReduceLROnPlateau reducing learning rate to 4.0960000478662555e-05.
191/191 [==============================] - 38s 199ms/step - loss: 0.4106 - accuracy: 0.8223 - val_loss: 0.4251 - val_accuracy: 0.8201 - lr: 5.1200e-05
Epoch 82/150
191/191 [==============================] - ETA: 0s - loss: 0.4111 - accuracy: 0.8271
Epoch 00082: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4111 - accuracy: 0.8271 - val_loss: 0.4250 - val_accuracy: 0.8194 - lr: 4.0960e-05
Epoch 83/150
191/191 [==============================] - ETA: 0s - loss: 0.4099 - accuracy: 0.8278
Epoch 00083: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4099 - accuracy: 0.8278 - val_loss: 0.4253 - val_accuracy: 0.8201 - lr: 4.0960e-05
Epoch 84/150
191/191 [==============================] - ETA: 0s - loss: 0.4056 - accuracy: 0.8268
Epoch 00084: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4056 - accuracy: 0.8268 - val_loss: 0.4250 - val_accuracy: 0.8207 - lr: 4.0960e-05
Epoch 85/150
191/191 [==============================] - ETA: 0s - loss: 0.4097 - accuracy: 0.8251
Epoch 00085: val_accuracy did not improve from 0.82141

Epoch 00085: ReduceLROnPlateau reducing learning rate to 3.2767999800853435e-05.
191/191 [==============================] - 38s 201ms/step - loss: 0.4097 - accuracy: 0.8251 - val_loss: 0.4252 - val_accuracy: 0.8201 - lr: 4.0960e-05
Epoch 86/150
191/191 [==============================] - ETA: 0s - loss: 0.4050 - accuracy: 0.8246
Epoch 00086: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4050 - accuracy: 0.8246 - val_loss: 0.4255 - val_accuracy: 0.8201 - lr: 3.2768e-05
Epoch 87/150
191/191 [==============================] - ETA: 0s - loss: 0.4154 - accuracy: 0.8207
Epoch 00087: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4154 - accuracy: 0.8207 - val_loss: 0.4240 - val_accuracy: 0.8201 - lr: 3.2768e-05
Epoch 88/150
191/191 [==============================] - ETA: 0s - loss: 0.4119 - accuracy: 0.8215
Epoch 00088: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 203ms/step - loss: 0.4119 - accuracy: 0.8215 - val_loss: 0.4229 - val_accuracy: 0.8201 - lr: 3.2768e-05
Epoch 89/150
191/191 [==============================] - ETA: 0s - loss: 0.4075 - accuracy: 0.8271
Epoch 00089: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4075 - accuracy: 0.8271 - val_loss: 0.4240 - val_accuracy: 0.8194 - lr: 3.2768e-05
Epoch 90/150
191/191 [==============================] - ETA: 0s - loss: 0.4044 - accuracy: 0.8263
Epoch 00090: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 202ms/step - loss: 0.4044 - accuracy: 0.8263 - val_loss: 0.4235 - val_accuracy: 0.8194 - lr: 3.2768e-05
Epoch 91/150
191/191 [==============================] - ETA: 0s - loss: 0.4049 - accuracy: 0.8315
Epoch 00091: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.4049 - accuracy: 0.8315 - val_loss: 0.4254 - val_accuracy: 0.8194 - lr: 3.2768e-05
Epoch 92/150
191/191 [==============================] - ETA: 0s - loss: 0.4061 - accuracy: 0.8263
Epoch 00092: val_accuracy did not improve from 0.82141

Epoch 00092: ReduceLROnPlateau reducing learning rate to 2.6214399258606137e-05.
191/191 [==============================] - 39s 202ms/step - loss: 0.4061 - accuracy: 0.8263 - val_loss: 0.4255 - val_accuracy: 0.8207 - lr: 3.2768e-05
Epoch 93/150
191/191 [==============================] - ETA: 0s - loss: 0.4094 - accuracy: 0.8253
Epoch 00093: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.4094 - accuracy: 0.8253 - val_loss: 0.4255 - val_accuracy: 0.8214 - lr: 2.6214e-05
Epoch 94/150
191/191 [==============================] - ETA: 0s - loss: 0.4079 - accuracy: 0.8271
Epoch 00094: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 203ms/step - loss: 0.4079 - accuracy: 0.8271 - val_loss: 0.4258 - val_accuracy: 0.8214 - lr: 2.6214e-05
Epoch 95/150
191/191 [==============================] - ETA: 0s - loss: 0.4080 - accuracy: 0.8246
Epoch 00095: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4080 - accuracy: 0.8246 - val_loss: 0.4253 - val_accuracy: 0.8207 - lr: 2.6214e-05
Epoch 96/150
191/191 [==============================] - ETA: 0s - loss: 0.4053 - accuracy: 0.8230
Epoch 00096: val_accuracy did not improve from 0.82141

Epoch 00096: ReduceLROnPlateau reducing learning rate to 2.09715188248083e-05.
191/191 [==============================] - 39s 204ms/step - loss: 0.4053 - accuracy: 0.8230 - val_loss: 0.4242 - val_accuracy: 0.8194 - lr: 2.6214e-05
Epoch 97/150
191/191 [==============================] - ETA: 0s - loss: 0.4060 - accuracy: 0.8253
Epoch 00097: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 202ms/step - loss: 0.4060 - accuracy: 0.8253 - val_loss: 0.4243 - val_accuracy: 0.8194 - lr: 2.0972e-05
Epoch 98/150
191/191 [==============================] - ETA: 0s - loss: 0.4018 - accuracy: 0.8269
Epoch 00098: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.4018 - accuracy: 0.8269 - val_loss: 0.4251 - val_accuracy: 0.8214 - lr: 2.0972e-05
Epoch 99/150
191/191 [==============================] - ETA: 0s - loss: 0.3958 - accuracy: 0.8333
Epoch 00099: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.3958 - accuracy: 0.8333 - val_loss: 0.4252 - val_accuracy: 0.8214 - lr: 2.0972e-05
Epoch 100/150
191/191 [==============================] - ETA: 0s - loss: 0.4085 - accuracy: 0.8230
Epoch 00100: val_accuracy did not improve from 0.82141

Epoch 00100: ReduceLROnPlateau reducing learning rate to 1.6777214477770033e-05.
191/191 [==============================] - 38s 200ms/step - loss: 0.4085 - accuracy: 0.8230 - val_loss: 0.4251 - val_accuracy: 0.8214 - lr: 2.0972e-05
Epoch 101/150
191/191 [==============================] - ETA: 0s - loss: 0.4029 - accuracy: 0.8266
Epoch 00101: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 202ms/step - loss: 0.4029 - accuracy: 0.8266 - val_loss: 0.4255 - val_accuracy: 0.8214 - lr: 1.6777e-05
Epoch 102/150
191/191 [==============================] - ETA: 0s - loss: 0.4038 - accuracy: 0.8236
Epoch 00102: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 204ms/step - loss: 0.4038 - accuracy: 0.8236 - val_loss: 0.4252 - val_accuracy: 0.8214 - lr: 1.6777e-05
Epoch 103/150
191/191 [==============================] - ETA: 0s - loss: 0.3995 - accuracy: 0.8310
Epoch 00103: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 201ms/step - loss: 0.3995 - accuracy: 0.8310 - val_loss: 0.4254 - val_accuracy: 0.8214 - lr: 1.6777e-05
Epoch 104/150
191/191 [==============================] - ETA: 0s - loss: 0.4049 - accuracy: 0.8289
Epoch 00104: val_accuracy did not improve from 0.82141

Epoch 00104: ReduceLROnPlateau reducing learning rate to 1.3421771291177721e-05.
191/191 [==============================] - 39s 205ms/step - loss: 0.4049 - accuracy: 0.8289 - val_loss: 0.4248 - val_accuracy: 0.8214 - lr: 1.6777e-05
Epoch 105/150
191/191 [==============================] - ETA: 0s - loss: 0.4030 - accuracy: 0.8266
Epoch 00105: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 202ms/step - loss: 0.4030 - accuracy: 0.8266 - val_loss: 0.4246 - val_accuracy: 0.8214 - lr: 1.3422e-05
Epoch 106/150
191/191 [==============================] - ETA: 0s - loss: 0.3972 - accuracy: 0.8351
Epoch 00106: val_accuracy did not improve from 0.82141
191/191 [==============================] - 39s 203ms/step - loss: 0.3972 - accuracy: 0.8351 - val_loss: 0.4255 - val_accuracy: 0.8207 - lr: 1.3422e-05
Epoch 107/150
191/191 [==============================] - ETA: 0s - loss: 0.4063 - accuracy: 0.8287
Epoch 00107: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4063 - accuracy: 0.8287 - val_loss: 0.4257 - val_accuracy: 0.8207 - lr: 1.3422e-05
Epoch 108/150
191/191 [==============================] - ETA: 0s - loss: 0.4074 - accuracy: 0.8274
Epoch 00108: val_accuracy did not improve from 0.82141

Epoch 00108: ReduceLROnPlateau reducing learning rate to 1.0737417323980481e-05.
191/191 [==============================] - 38s 199ms/step - loss: 0.4074 - accuracy: 0.8274 - val_loss: 0.4258 - val_accuracy: 0.8201 - lr: 1.3422e-05
Epoch 109/150
191/191 [==============================] - ETA: 0s - loss: 0.4093 - accuracy: 0.8266
Epoch 00109: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4093 - accuracy: 0.8266 - val_loss: 0.4253 - val_accuracy: 0.8207 - lr: 1.0737e-05
Epoch 110/150
191/191 [==============================] - ETA: 0s - loss: 0.4041 - accuracy: 0.8322
Epoch 00110: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.4041 - accuracy: 0.8322 - val_loss: 0.4254 - val_accuracy: 0.8207 - lr: 1.0737e-05
Epoch 111/150
191/191 [==============================] - ETA: 0s - loss: 0.4014 - accuracy: 0.8284
Epoch 00111: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4014 - accuracy: 0.8284 - val_loss: 0.4255 - val_accuracy: 0.8207 - lr: 1.0737e-05
Epoch 112/150
191/191 [==============================] - ETA: 0s - loss: 0.4076 - accuracy: 0.8261
Epoch 00112: val_accuracy did not improve from 0.82141

Epoch 00112: ReduceLROnPlateau reducing learning rate to 8.589933713665232e-06.
191/191 [==============================] - 38s 200ms/step - loss: 0.4076 - accuracy: 0.8261 - val_loss: 0.4255 - val_accuracy: 0.8207 - lr: 1.0737e-05
Epoch 113/150
191/191 [==============================] - ETA: 0s - loss: 0.4020 - accuracy: 0.8278
Epoch 00113: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4020 - accuracy: 0.8278 - val_loss: 0.4257 - val_accuracy: 0.8207 - lr: 8.5899e-06
Epoch 114/150
191/191 [==============================] - ETA: 0s - loss: 0.3999 - accuracy: 0.8332
Epoch 00114: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.3999 - accuracy: 0.8332 - val_loss: 0.4257 - val_accuracy: 0.8207 - lr: 8.5899e-06
Epoch 115/150
191/191 [==============================] - ETA: 0s - loss: 0.4025 - accuracy: 0.8315
Epoch 00115: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.4025 - accuracy: 0.8315 - val_loss: 0.4258 - val_accuracy: 0.8207 - lr: 8.5899e-06
Epoch 116/150
191/191 [==============================] - ETA: 0s - loss: 0.4041 - accuracy: 0.8269
Epoch 00116: val_accuracy did not improve from 0.82141

Epoch 00116: ReduceLROnPlateau reducing learning rate to 6.871946970932186e-06.
191/191 [==============================] - 38s 197ms/step - loss: 0.4041 - accuracy: 0.8269 - val_loss: 0.4257 - val_accuracy: 0.8207 - lr: 8.5899e-06
Epoch 117/150
191/191 [==============================] - ETA: 0s - loss: 0.4059 - accuracy: 0.8263
Epoch 00117: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4059 - accuracy: 0.8263 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 6.8719e-06
Epoch 118/150
191/191 [==============================] - ETA: 0s - loss: 0.3994 - accuracy: 0.8348
Epoch 00118: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.3994 - accuracy: 0.8348 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 6.8719e-06
Epoch 119/150
191/191 [==============================] - ETA: 0s - loss: 0.3988 - accuracy: 0.8299
Epoch 00119: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.3988 - accuracy: 0.8299 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 6.8719e-06
Epoch 120/150
191/191 [==============================] - ETA: 0s - loss: 0.4023 - accuracy: 0.8317
Epoch 00120: val_accuracy did not improve from 0.82141

Epoch 00120: ReduceLROnPlateau reducing learning rate to 5.497557503986173e-06.
191/191 [==============================] - 38s 199ms/step - loss: 0.4023 - accuracy: 0.8317 - val_loss: 0.4257 - val_accuracy: 0.8207 - lr: 6.8719e-06
Epoch 121/150
191/191 [==============================] - ETA: 0s - loss: 0.4010 - accuracy: 0.8317
Epoch 00121: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.4010 - accuracy: 0.8317 - val_loss: 0.4258 - val_accuracy: 0.8201 - lr: 5.4976e-06
Epoch 122/150
191/191 [==============================] - ETA: 0s - loss: 0.3939 - accuracy: 0.8340
Epoch 00122: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 196ms/step - loss: 0.3939 - accuracy: 0.8340 - val_loss: 0.4257 - val_accuracy: 0.8201 - lr: 5.4976e-06
Epoch 123/150
191/191 [==============================] - ETA: 0s - loss: 0.3997 - accuracy: 0.8255
Epoch 00123: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 196ms/step - loss: 0.3997 - accuracy: 0.8255 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 5.4976e-06
Epoch 124/150
191/191 [==============================] - ETA: 0s - loss: 0.4062 - accuracy: 0.8274
Epoch 00124: val_accuracy did not improve from 0.82141

Epoch 00124: ReduceLROnPlateau reducing learning rate to 4.398046075948514e-06.
191/191 [==============================] - 38s 197ms/step - loss: 0.4062 - accuracy: 0.8274 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 5.4976e-06
Epoch 125/150
191/191 [==============================] - ETA: 0s - loss: 0.4072 - accuracy: 0.8268
Epoch 00125: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.4072 - accuracy: 0.8268 - val_loss: 0.4258 - val_accuracy: 0.8201 - lr: 4.3980e-06
Epoch 126/150
191/191 [==============================] - ETA: 0s - loss: 0.4008 - accuracy: 0.8258
Epoch 00126: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.4008 - accuracy: 0.8258 - val_loss: 0.4258 - val_accuracy: 0.8201 - lr: 4.3980e-06
Epoch 127/150
191/191 [==============================] - ETA: 0s - loss: 0.4023 - accuracy: 0.8299
Epoch 00127: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.4023 - accuracy: 0.8299 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 4.3980e-06
Epoch 128/150
191/191 [==============================] - ETA: 0s - loss: 0.4048 - accuracy: 0.8245
Epoch 00128: val_accuracy did not improve from 0.82141

Epoch 00128: ReduceLROnPlateau reducing learning rate to 3.518437006277964e-06.
191/191 [==============================] - 38s 200ms/step - loss: 0.4048 - accuracy: 0.8245 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 4.3980e-06
Epoch 129/150
191/191 [==============================] - ETA: 0s - loss: 0.4049 - accuracy: 0.8309
Epoch 00129: val_accuracy did not improve from 0.82141
191/191 [==============================] - 37s 196ms/step - loss: 0.4049 - accuracy: 0.8309 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 3.5184e-06
Epoch 130/150
191/191 [==============================] - ETA: 0s - loss: 0.3982 - accuracy: 0.8314
Epoch 00130: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.3982 - accuracy: 0.8314 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 3.5184e-06
Epoch 131/150
191/191 [==============================] - ETA: 0s - loss: 0.3952 - accuracy: 0.8325
Epoch 00131: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 196ms/step - loss: 0.3952 - accuracy: 0.8325 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 3.5184e-06
Epoch 132/150
191/191 [==============================] - ETA: 0s - loss: 0.3937 - accuracy: 0.8302
Epoch 00132: val_accuracy did not improve from 0.82141

Epoch 00132: ReduceLROnPlateau reducing learning rate to 2.814749677781947e-06.
191/191 [==============================] - 38s 198ms/step - loss: 0.3937 - accuracy: 0.8302 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 3.5184e-06
Epoch 133/150
191/191 [==============================] - ETA: 0s - loss: 0.4040 - accuracy: 0.8264
Epoch 00133: val_accuracy did not improve from 0.82141
191/191 [==============================] - 37s 196ms/step - loss: 0.4040 - accuracy: 0.8264 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 2.8147e-06
Epoch 134/150
191/191 [==============================] - ETA: 0s - loss: 0.4053 - accuracy: 0.8271
Epoch 00134: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 196ms/step - loss: 0.4053 - accuracy: 0.8271 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 2.8147e-06
Epoch 135/150
191/191 [==============================] - ETA: 0s - loss: 0.3970 - accuracy: 0.8289
Epoch 00135: val_accuracy did not improve from 0.82141
191/191 [==============================] - 37s 196ms/step - loss: 0.3970 - accuracy: 0.8289 - val_loss: 0.4261 - val_accuracy: 0.8201 - lr: 2.8147e-06
Epoch 136/150
191/191 [==============================] - ETA: 0s - loss: 0.4072 - accuracy: 0.8251
Epoch 00136: val_accuracy did not improve from 0.82141

Epoch 00136: ReduceLROnPlateau reducing learning rate to 2.2517997422255576e-06.
191/191 [==============================] - 38s 200ms/step - loss: 0.4072 - accuracy: 0.8251 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 2.8147e-06
Epoch 137/150
191/191 [==============================] - ETA: 0s - loss: 0.4029 - accuracy: 0.8289
Epoch 00137: val_accuracy did not improve from 0.82141
191/191 [==============================] - 37s 196ms/step - loss: 0.4029 - accuracy: 0.8289 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 2.2518e-06
Epoch 138/150
191/191 [==============================] - ETA: 0s - loss: 0.3993 - accuracy: 0.8297
Epoch 00138: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.3993 - accuracy: 0.8297 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 2.2518e-06
Epoch 139/150
191/191 [==============================] - ETA: 0s - loss: 0.4020 - accuracy: 0.8300
Epoch 00139: val_accuracy did not improve from 0.82141
191/191 [==============================] - 37s 196ms/step - loss: 0.4020 - accuracy: 0.8300 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 2.2518e-06
Epoch 140/150
191/191 [==============================] - ETA: 0s - loss: 0.4048 - accuracy: 0.8278
Epoch 00140: val_accuracy did not improve from 0.82141

Epoch 00140: ReduceLROnPlateau reducing learning rate to 1.801439793780446e-06.
191/191 [==============================] - 38s 197ms/step - loss: 0.4048 - accuracy: 0.8278 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 2.2518e-06
Epoch 141/150
191/191 [==============================] - ETA: 0s - loss: 0.4024 - accuracy: 0.8266
Epoch 00141: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4024 - accuracy: 0.8266 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.8014e-06
Epoch 142/150
191/191 [==============================] - ETA: 0s - loss: 0.3874 - accuracy: 0.8356
Epoch 00142: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.3874 - accuracy: 0.8356 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.8014e-06
Epoch 143/150
191/191 [==============================] - ETA: 0s - loss: 0.3974 - accuracy: 0.8305
Epoch 00143: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.3974 - accuracy: 0.8305 - val_loss: 0.4260 - val_accuracy: 0.8201 - lr: 1.8014e-06
Epoch 144/150
191/191 [==============================] - ETA: 0s - loss: 0.3966 - accuracy: 0.8310
Epoch 00144: val_accuracy did not improve from 0.82141

Epoch 00144: ReduceLROnPlateau reducing learning rate to 1.441151835024357e-06.
191/191 [==============================] - 38s 197ms/step - loss: 0.3966 - accuracy: 0.8310 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.8014e-06
Epoch 145/150
191/191 [==============================] - ETA: 0s - loss: 0.4042 - accuracy: 0.8233
Epoch 00145: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 200ms/step - loss: 0.4042 - accuracy: 0.8233 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.4412e-06
Epoch 146/150
191/191 [==============================] - ETA: 0s - loss: 0.4038 - accuracy: 0.8256
Epoch 00146: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.4038 - accuracy: 0.8256 - val_loss: 0.4258 - val_accuracy: 0.8201 - lr: 1.4412e-06
Epoch 147/150
191/191 [==============================] - ETA: 0s - loss: 0.4104 - accuracy: 0.8230
Epoch 00147: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 199ms/step - loss: 0.4104 - accuracy: 0.8230 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.4412e-06
Epoch 148/150
191/191 [==============================] - ETA: 0s - loss: 0.4055 - accuracy: 0.8276
Epoch 00148: val_accuracy did not improve from 0.82141

Epoch 00148: ReduceLROnPlateau reducing learning rate to 1.1529215043992736e-06.
191/191 [==============================] - 37s 196ms/step - loss: 0.4055 - accuracy: 0.8276 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.4412e-06
Epoch 149/150
191/191 [==============================] - ETA: 0s - loss: 0.3949 - accuracy: 0.8300
Epoch 00149: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 198ms/step - loss: 0.3949 - accuracy: 0.8300 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.1529e-06
Epoch 150/150
191/191 [==============================] - ETA: 0s - loss: 0.3942 - accuracy: 0.8338
Epoch 00150: val_accuracy did not improve from 0.82141
191/191 [==============================] - 38s 197ms/step - loss: 0.3942 - accuracy: 0.8338 - val_loss: 0.4259 - val_accuracy: 0.8201 - lr: 1.1529e-06


# In[ ]:


plots_visual(history)


# PREDICT MODEL

# In[ ]:


from tensorflow.keras.models import load_model
model  = load_model('/content/drive/My Drive/kaggle/best_model.h5')


# In[ ]:


test =  pd.read_csv('/content/drive/My Drive/kaggle/test (1).csv')

def prediction(data, namefile):
    data = data.copy()
    data['text'] = data['text'].apply(clean_text)
    dataval = tokenizer.texts_to_sequences(data['text'])
    text_sequens = pad_sequences(dataval, maxlen=maxLen,padding='post')
    predictions = model.predict(text_sequens)
    predictions = np.round(predictions).astype(int).reshape(3263)
    testData = data[['id']]
    testData['target'] = predictions
    testData.to_csv(f'{namefile}.csv',index=False)


# In[ ]:


prediction(test, 'submitnew')

