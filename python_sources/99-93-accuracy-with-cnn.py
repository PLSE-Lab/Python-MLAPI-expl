#!/usr/bin/env python
# coding: utf-8

# **For a detailed description of this work see the Overview of the source code: https://bitbucket.org/govindnh4cl/snake_eyes/src/master/Readme.md**
# 
# I was worried about the timeout in kaggle, so I am only running only 1 epochs here. Because of which the accuracy reported at the end is lower. On my local setup I ran 20 epochs (around 20 minutes on GTX960) and got the reported 99.93% score. I couldn't figure out a way to keep pre-trained network in kaggle even though its size is < 1MB. So, I have kept it in the repository above. Just clone/download the repository and run to validate the accuracy score. 
# 
# 

# In[8]:


import os
import numpy as np
import tensorflow as tf

my_random_seed = 1331
np.random.seed(my_random_seed)
tf.set_random_seed(my_random_seed)

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[ ]:


'''
Configurable parameters
'''
model_path = 'model_test.h5'
in_dir = '../input/snake-eyes'
batch_size = 512
epochs = 1


# In[10]:


'''
Load data into memory
'''
def _read_vectors(filename):
    return np.fromfile(filename, dtype=np.uint8).reshape(-1, 401)


# ------ Setup training and validation sets ----------
snk = np.vstack(tuple(_read_vectors(
    os.path.join(in_dir, "snakeeyes_{:02d}.dat".format(nn))) for nn in range(10)))

x = snk[:, 1:]
y_ = snk[:, 0]

# Convert labels to one hot
y = np.zeros([len(y_), 12], dtype=bool)
for i in range(len(y_)):
    label = y_[i]
    y[i, label - 1] = True

x_train, x_validation, y_train, y_validation =     train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

mean = x_train.mean(axis=0)  # Compute data statistics on training data only
std = x_train.std(axis=0)

# Normalize data to be zero mean and unit variance
x_train = (x_train - mean)/std
x_validation = (x_validation - mean)/std

# Reshape
x_train = x_train.reshape(-1, 20, 20, 1)
x_validation = x_validation.reshape(-1, 20, 20, 1)

# ------ Setup test set ----------
snk_test = _read_vectors(os.path.join(in_dir, "snakeeyes_test.dat"))
x_test = snk_test[:, 1:]
y_test_ = snk_test[:, 0]

# Convert labels to one hot
y_test = np.zeros([len(y_test_), 12], dtype=bool)
for i in range(len(y_test)):
    label = y_test_[i]
    y_test[i, label - 1] = True

x_test = (x_test - mean)/std  # Normalize
x_test = x_test.reshape(-1, 20, 20, 1)  # Reshape


# In[ ]:


'''
Get Model instance
''' 
# Define model
inputs = Input(shape=(20, 20, 1))
x = Conv2D(2, 5, padding='same', activation='relu')(inputs)
x = Conv2D(8, 5, padding='same', activation='relu')(x)
x = Conv2D(8, 5, padding='same', activation='relu')(x)
x = Conv2D(8, 5, padding='same', activation='relu')(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(12, activation='softmax')(x)
outputs = x

m = Model(inputs=inputs, outputs=outputs)
print(m.summary())    

opt = RMSprop(lr=0.001)
m.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')
cb_chkpt = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True)

m.fit(x=x_train, y=y_train,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=(x_validation, y_validation),
           callbacks=[cb_chkpt])    


# In[ ]:


'''
Evaluate model performance
'''
test_loss, test_accuracy = m.evaluate(x_test, y_test)
print('Test loss: {:} Test Accuracy: {:}'.format(test_loss, test_accuracy))


# In[ ]:




