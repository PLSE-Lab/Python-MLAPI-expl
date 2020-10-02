#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical


# In[ ]:


# Input data files are available in the "/kaggle/input/108-1-dl-app-hw1/" directory.
DATA_DIR = '/kaggle/input/108-1-dl-app-hw1/'
TRAIN_DATA_FILE = DATA_DIR + 'emnist-byclass-train.npz'
TEST_DATA_FILE = DATA_DIR + 'emnist-byclass-test.npz'


# In[ ]:


# Load training data
data = np.load(TRAIN_DATA_FILE)
train_labels = data['training_labels']
train_images = data['training_images']


# In[ ]:


trn_images = train_images.reshape((train_images.shape[0], 28 * 28))
trn_images = trn_images.astype('float32') / 255
trn_labels = to_categorical(train_labels)


# In[ ]:


# Define Your Own Network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(256, activation='relu', input_shape=(512,)))
network.add(layers.Dense(128, activation='relu', input_shape=(256,)))
network.add(layers.Dense(62, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


network.fit(trn_images, trn_labels, epochs=5, batch_size=128)


# In[ ]:


# Evalute our model on test data
test_images = np.load(TEST_DATA_FILE)['testing_images']


# In[ ]:


tst_images = test_images.reshape((test_images.shape[0], 28 * 28))
tst_images = tst_images.astype('float32') / 255


# In[ ]:


results = network.predict_classes(tst_images)
results


# In[ ]:


# Print results in CSV format and upload to Kaggle
with open('pred_results.csv', 'w') as f:
    f.write('Id,Category\n')
    for i in range(len(results)):
        f.write(str(i) + ',' + str(results[i]) + '\n')


# In[ ]:


# Download your results!
from IPython.display import FileLink
FileLink('pred_results.csv')

