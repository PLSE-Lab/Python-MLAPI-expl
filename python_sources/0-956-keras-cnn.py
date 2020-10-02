#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


HEIGHT = WIDTH = 28
CHANNELS = 1

BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHES = 50


# In[3]:


df_origin = pd.read_csv('../input/fashion-mnist_train.csv', header = None)


# In[4]:


labels = df_origin.iloc[1:, 0].astype(np.int).values
images = df_origin.iloc[1:, 1:].astype(np.float).values


# ### Scale and Normalization of Input Data

# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

images = StandardScaler().fit_transform(images)


# In[6]:


images = Normalizer().fit_transform(images)


# ### Reshap Input data and One hot labels

# In[7]:


from keras.utils import to_categorical
images = images.reshape((images.shape[0], HEIGHT, WIDTH, CHANNELS))
labels = to_categorical(labels, num_classes=NUM_CLASSES)


# ### Split train, dev dataset

# In[8]:


X_train, X_dev, y_train, y_dev = train_test_split(images, labels, test_size=0.2)
print(X_train.shape, X_dev.shape, y_train.shape, y_dev.shape)


# ### Model

# In[9]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# In[10]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(HEIGHT, WIDTH, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))


# In[11]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[25]:


#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#%matplotlib inline
#model_to_dot(model).create(prog='dot', format='png')
model.summary()


# ### Visualization

# In[28]:


from keras.callbacks import Callback
from IPython.display import clear_output

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show();


# ### Train

# In[29]:


history = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHES,
          verbose=1,
          callbacks=[PlotLearning()],
          validation_data=(X_dev, y_dev))
score = model.evaluate(X_dev, y_dev, verbose=0)

