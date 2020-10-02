#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import time
import json
from glob import glob
from PIL import Image

import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import pickle


# In[ ]:


folder = "../input/pictest/"
images = os.listdir(folder)
image_model = InceptionV3(weights='imagenet')
model_new = tf.keras.Model(image_model.input, image_model.layers[-2].output)


# In[ ]:


img_features = dict()
import time
start_time = time.time()
for img in images:
    img1 = image.load_img(folder + img, target_size=(299, 299, 3))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    fea_x = model_new.predict(x)
    fea_x1 = np.reshape(fea_x , fea_x.shape[1])
    img_features[img] = fea_x1
print("--- %s seconds for feature extraction ---" % (time.time() - start_time))


# In[ ]:


images_new = os.listdir("../input/new-pictest/")
for img in images_new:
    img1 = image.load_img("../input/new-pictest/" + img, target_size=(299, 299, 3))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    fea_x = model_new.predict(x)
    fea_x1 = np.reshape(fea_x , fea_x.shape[1])
    img_features[img] = fea_x1
print("--- %s seconds for feature extraction ---" % (time.time() - start_time))


# In[ ]:


from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


model = load_model('../input/model-512-6-50/model_512_6_50.h5')


# In[ ]:


#Read Word Index
infile = open("../input/flickr8k-captions/word_index.pkl",'rb')
word_index = pickle.load(infile)
infile.close()

infile = open("../input/flickr8k-captions/index_word.pkl",'rb')
index_word = pickle.load(infile)
infile.close()


# In[ ]:


def createCaption(photo, model, max_length = 34):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_index[w] for w in in_text.split() if w in word_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    return final


# In[ ]:


preds = dict()
for p in img_features.keys():
    
    sample_fea = img_features[p]
#     if p in ['11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg']:
#         x=plt.imread("../input/new-pictest/" + p)
#     else:
#         x=plt.imread(folder + p)
#     plt.imshow(x)
#     plt.show()

            
    a = createCaption((sample_fea).reshape((1,2048)), model)
    print(' '.join(a))
    preds[p] = ' '.join(a)
    


# In[ ]:


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 


# In[ ]:


tokenizer = Tokenizer()
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt     -O /tmp/sonnets.txt')
data = open('/tmp/sonnets.txt').read()

corpus = data.lower().split("\n")


# In[ ]:


tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# In[ ]:


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)


# In[ ]:


model = Sequential()
model.add(Embedding(total_words, 150, input_length=max_sequence_len-1))# Your Embedding Layer
model.add(Bidirectional(LSTM(512, return_sequences = True)))# An LSTM Layer
model.add(Dropout(0.5))# A dropout layer)
model.add(LSTM(128))# Another LSTM Layer)
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.05)))# A Dense Layer including regularizers)
model.add(Dense(total_words, activation='softmax'))# A Dense Layer)
# Pick an optimizer
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])# Pick a loss function and an optimizer
print(model.summary())


# In[ ]:


history = model.fit(predictors, label, epochs=100, verbose=1)


# In[ ]:


for p in preds.keys():
       
    if p in ['11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg']:
        x=plt.imread("../input/new-pictest/" + p)
    else:
        x=plt.imread(folder + p)
    plt.imshow(x)
    plt.show()

    seed_text = preds[p]
    
    next_words = 100
  
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)


# In[ ]:


# #256 unit LSTM, with dropout 0.20
# model_3_20_256 = load_model('../input/model-256-x-20/model_256_3_20.h5')
# model_6_20_256 = load_model('../input/model-256-x-20/model_256_6_20.h5')
# model_10_20_256 = load_model('../input/model-256-x-20/model_256_10_20.h5')

# #256 unit LSTM, with dropout 0,50
# model_3_50_256 = load_model('../input/model-256-x-50/model_256_3_50.h5') 
# model_6_50_256 = load_model('../input/model-256-x-50/model_256_6_50.h5')
# model_10_50_256 = load_model('../input/model-256-x-50/model_256_10_50.h5')
# model_20_50_256 = load_model('../input/model-256-x-50/model_256_20_50.h5') 

# #512 unit LSTM, with dropout 0.20
# model_3_20_512 = load_model('../input/model-512-x-20/model_512_3_20.h5')
# model_6_20_512 = load_model('../input/model-512-x-20/model_512_6_20.h5')
# model_10_20_512 = load_model('../input/model-512-x-20/model_512_10_20.h5')

# ##256 unit LSTM, with dropout 0.50
# model_3_50_512 = load_model('../input/model-512-6-50/model_512_3_50.h5')
# model_6_50_512 = load_model('../input/model-512-6-50/model_512_6_50.h5')
# model_10_50_512 = load_model('../input/model-512-6-50/model_512_10_50.h5')

# model_256_20 = [model_3_20_256, model_6_20_256, model_10_20_256]
# model_256_50 = [model_3_50_256, model_6_50_256, model_10_50_256, model_20_50_256]
# model_512_20 = [model_3_20_512, model_6_20_512, model_10_20_512]  
# model_512_50 = [model_3_50_512, model_6_50_512, model_10_50_512]


# In[ ]:


# for p in img_features.keys():
    
#     sample_fea = img_features[p]
#     if p in ['11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg']:
#         x=plt.imread("../input/new-pictest/" + p)
#     else:
#         x=plt.imread(folder + p)
#     plt.imshow(x)
#     plt.show()

#     conf = ""
#     print("256 unit LSTM, with dropout 0.20")
#     for i, model in enumerate(model_256_20):
#         if i == 0:
#             conf = "Picture/batch = 3: "
#         elif i == 1:
#             conf = "Picture/batch = 6: "
#         else:
#             conf = "Picture/batch = 10: "    
            
#         a = createCaption((sample_fea).reshape((1,2048)), model)
#         print(conf, ' '.join(a))
        
#     print("\n256 unit LSTM, with dropout 0.50")
#     for i, model in enumerate(model_256_50):
#         if i == 0:
#             conf = "Picture/batch = 3: "
#         elif i == 1:
#             conf = "Picture/batch = 6: "
#         elif i == 2:
#             conf = "Picture/batch = 10: "
#         else:
#             conf = "Picture/batch = 20: "
            
#         a = createCaption((sample_fea).reshape((1,2048)), model)
#         print(conf, ' '.join(a))
        
#     print("\n512 unit LSTM, with dropout 0.20")
#     for i, model in enumerate(model_512_20):
#         if i == 0:
#             conf = "Picture/batch = 3: "
#         elif i == 1:
#             conf = "Picture/batch = 6: "
#         else:
#             conf = "Picture/batch = 10: "
            
#         a = createCaption((sample_fea).reshape((1,2048)), model)
#         print(conf, ' '.join(a))
        
#     print("\n512 unit LSTM, with dropout 0.50")
#     for i, model in enumerate(model_512_50):
#         if i == 0:
#             conf = "Picture/batch = 3: "
#         elif i == 1:
#             conf = "Picture/batch = 6: "
#         else:
#             conf = "Picture/batch = 10: "
            
#         a = createCaption((sample_fea).reshape((1,2048)), model)
#         print(conf, ' '.join(a))

