#!/usr/bin/env python
# coding: utf-8

# # **Character-Level Classification using CNNs**
# 
# For general background on this topic, check out the following link.
# 
# [Best Practices for Document Classifcation with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)
# 
# I am implementing the network described in this [paper](https://arxiv.org/pdf/1606.01781.pdf) which was also done by someone else in [this kernel](https://www.kaggle.com/robwec/character-level-author-identification-with-cnns) which I suggest you also check out. I found [this kernel](https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights) utilizing a CNN approach helpful as well. 
# 
# Overall, this is most likely not the best approach for this particular dataset but may be of use for others in the future tackling larger datasets. The authors in the paper linked above describe how this method does better on larger datasets. 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


x_train = train.iloc[:,1].values
y_train = train.iloc[:,2].values


# The following block of code is adapted from [this repository](https://github.com/johnb30/py_crepe).
# 
# This will take the sentences as input and turn each of them into a sparse character array. 
# 
# The max length is equivalent to the length of the sentence. 250 should be more than enough and you most likely can go even lower. 

# In[ ]:


import string

maxlen = 250
alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
vocab_size = len(alphabet)
check = set(alphabet)

vocab = {}
reverse_vocab = {}
for ix, t in enumerate(alphabet):
    vocab[t] = ix
    reverse_vocab[ix] = t

input_array = np.zeros((len(x_train), maxlen, vocab_size))
for i, sentence in enumerate(x_train):
    counter = 0
    sentence_array = np.zeros((maxlen, vocab_size))
    chars = list(sentence.lower().replace(' ', ''))
    for c in chars:
        if counter >= maxlen:
            pass
        else:
            char_array = np.zeros(vocab_size, dtype=np.int)
            if c in check:
                ix = vocab[c]
                char_array[ix] = 1
            sentence_array[counter, :] = char_array
            counter +=1
    input_array[i, :, :] = sentence_array


# In[ ]:


print(np.shape(input_array))


# Following is a One Hot Encoding of the labels (the authors)

# In[ ]:


from sklearn.preprocessing import LabelBinarizer

one_hot = LabelBinarizer()
y_train = one_hot.fit_transform(y_train)
y_train


# The following is the Keras architecture described in the paper linked to in the beginning. They describe 9-layer, 17-layer, 29-layer, and 49-layer variations but they all proceed in the same general manner as below. 
# 
# Do note that this method utilizes batch normalization instead of dropout. 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(250, 69)))
model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()


# In[ ]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(input_array, y_train, validation_split=0.2, epochs=40, batch_size=64, verbose=2)


# Data Preparation of the Test Set

# In[ ]:


x_test = test.iloc[:,1].values


# In[ ]:


test_array = np.zeros((len(x_test), maxlen, vocab_size))
for i, sentence in enumerate(x_test):
    counter = 0
    sentence_array = np.zeros((maxlen, vocab_size))
    chars = list(sentence.lower().replace(' ', ''))
    for c in chars:
        if counter >= maxlen:
            pass
        else:
            char_array = np.zeros(vocab_size, dtype=np.int)
            if c in check:
                ix = vocab[c]
                char_array[ix] = 1
            sentence_array[counter, :] = char_array
            counter +=1
    test_array[i, :, :] = sentence_array


# In[ ]:


print(np.shape(test_array))


# In[ ]:


y_test = model.predict_proba(test_array)


# In[ ]:


ids = test['id']


# In[ ]:


submission = pd.DataFrame(y_test, columns=['EAP', 'HPL', 'MWS'])
submission.insert(0, "id", ids)
submission.to_csv("submission.csv", index=False)

