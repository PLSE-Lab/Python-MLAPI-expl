#!/usr/bin/env python
# coding: utf-8

# In this notebook i will show how to build a very simple convolutional neuronal network using Keras to classify single characters extracted from the captcha dataset images <br/>
# You can find more code & notebooks in my repository: https://github.com/Vykstorm/CaptchaDL

# ## Import statements

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import sys, os

import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from warnings import filterwarnings

filterwarnings('ignore')


# ## Import data

# Here i use a generated dataset which consists of 50k samples of single char images of size 40x40 pixels extracted from the captcha dataset (as indicated in the part 2 notebook)

# In[ ]:


data = np.load('/kaggle/input/captchadlutils/repository/Vykstorm-CaptchaDL-utils-a8458b5/chars.npz')
X, y = data['X'], data['y']
alphabet = np.load('/kaggle/input/reading-captcha-dataset-part-1/preprocessed-data.npz')['alphabet']


# In[ ]:


X.shape[0]


# In[ ]:


indices = np.random.choice(np.arange(0, X.shape[0]), size=10)
X_batch, y_batch = X[indices], y[indices]

fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for i, j in product(range(0, 2), range(0, 5)):
    k = i * 3 + j
    plt.sca(ax[i, j])
    plt.imshow(X_batch[k, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('"{}"'.format(alphabet[y_batch[k].argmax()]))


# ## Build the model

# We will build a deep learning model with the next layers: <br/>
# * (Convolutional layers + Max Pool layers) * 3
# * Flatten layer to convert image features into a vector
# * 2 dense layers (the last one with softmax activation and 19 units) 

# In[ ]:


num_classes = len(alphabet)

t_in = Input(shape=(X.shape[1:]))

x = t_in

x = Conv2D(32, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(x)
x = MaxPool2D((2, 2))(x)

x = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)

x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)

x = Flatten()(x)
x = Dense(96, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

t_out = x
model = Model([t_in], [t_out])


# In[ ]:


model.summary()


# 
# Compile the model

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ## Split the dataset in train & test sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, stratify=y.argmax(axis=1))


# ## Train the model

# In[ ]:


epochs = 8
callbacks = [
    EarlyStopping(min_delta=0.01, monitor='val_loss', mode='min', patience=1),
    ModelCheckpoint('weights.hdf5', monitor='val_loss', save_weights_only=True, mode='min')
]
result = model.fit(X_train, y_train, batch_size=8, epochs=epochs, verbose=True, validation_split=0.1, callbacks=callbacks)
history = result.history


# Print performance scores during the training phase

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(11, 4))

plt.sca(ax[0])
plt.plot(history['loss'], color='red')
plt.plot(history['val_loss'], color='blue')
plt.legend(['Loss', 'Val. Loss'])
plt.xlabel('Epoch')
plt.title('Loss')
plt.tight_layout()

plt.sca(ax[1])
plt.plot(history['acc'], color='red')
plt.plot(history['val_acc'], color='blue')
plt.legend(['Accuracy', 'Val. Accuracy'])
plt.xlabel('Epoch')
plt.title('Accuracy')

plt.suptitle('Model performance on training')

plt.tight_layout()
plt.subplots_adjust(top=0.85)


# Set the weights which gave the model lower validation loss in the training phase

# In[ ]:


model.load_weights('weights.hdf5')


# ## Evaluate the model

# Evaluate the model on the test set

# In[ ]:


y_test_pred = model.predict(X_test, verbose=True)
y_test_labels = y_test.argmax(axis=1)
y_test_labels_pred = y_test_pred.argmax(axis=1)


# In[ ]:


print('Accuracy on test set: {}'.format(np.round(accuracy_score(y_test_labels, y_test_labels_pred), 4)))


# The next graphic shows the confusion matrix of the predictions

# In[ ]:


plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test_labels, y_test_labels_pred), annot=True, fmt='d',
                            xticklabels=alphabet, yticklabels=alphabet)
plt.title('Confusion matrix of eval predictions');


# The next cell prints different metric scores of our predictions (for each target class)

# In[ ]:


print(classification_report(y_test_labels, y_test_labels_pred, target_names=alphabet))


# ## Conclusions

# If you check the 'precision' score and the confusion matrix graphic, you can see that the worst results are raised when we predict 'm' is an 'n'. This is because of 'm' being cut sometimes when we split the image and is transformed to an 'n'. That means for that samples the label is wrong

# Despite that, we can see that the model generalizes well and shows good results (accuracy over 0.9) with only a few training iterations

# I saved the model weights in https://github.com/Vykstorm/CaptchaDL/blob/master/models/.char-classifier-weights.hdf5?raw=true so you can download it and build the model without training <br/>
# Set the model weights using model.load_weights()
