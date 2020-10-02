#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# In[ ]:


import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
import matplotlib.pyplot as plt
dict1= unpickle("../input/data_batch_1")
dicttest=unpickle("../input/test_batch")
# load (downloaded if needed) the MNIST dataset
(X_train, y_train)= dict1[b'data'], dict1[b'labels']
(X_test, y_test)= dicttest[b'data'], dicttest[b'labels']

X_train=X_train[:7500]
X_test=X_test[:2500]
y_test=y_test[:2500]
y_train=y_train[:7500]


# In[ ]:


seed = 7
numpy.random.seed(seed)
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')


# In[ ]:


X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[ ]:


def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(48, (3, 3), input_shape=(3, 32, 32), activation='relu'))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (3, 3), border_mode='same', activation='relu'))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(192, (3, 3), border_mode='same', activation='relu'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# In[ ]:


# In[ ]:


# build the model
model = larger_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
#model.fit(X_train_gray, y_train, validation_data=(X_test_gray, y_test), epochs=10, batch_size=200, verbose=2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


# In[ ]:


import sklearn
from sklearn.metrics import confusion_matrix


# In[ ]:


predictions = model.predict(X_test)

y_test_ = []
pred = []

for x in y_test:
    y_test_.append(numpy.argmax(x))

for x in predictions:
    pred.append(numpy.argmax(x))


#matrice confusion
sklearn.metrics.confusion_matrix(y_test_,pred, labels= None, sample_weight= None)


# In[ ]:


from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score(y_test_,pred)

