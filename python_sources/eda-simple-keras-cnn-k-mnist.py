#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis

# In[ ]:


import time

import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE


# In[ ]:


print(os.listdir('../input'))


# Note that .npz files are stored as a dictionary, therefore we need to access its keys to retrieve the numpy array. In this case, it is `'arr_0'`.

# In[ ]:


train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']
train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']


# ### Preprocess the Data

# In[ ]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)


# We first scale the dataset.

# In[ ]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# Display some sample images to get an idea of the distribution

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])


# ### Visualization

# Let's visualize the dataset using T-SNE. The algorithm will take a minute or two to run.

# In[ ]:


# Sample from the training set
sample_size = 8000

np.random.seed(2018)
idx = np.random.choice(60000, size=sample_size, replace=False)
train_sample = train_images.reshape(60000, -1)[idx, :]
label_sample = train_labels[idx]

# Generate 2D embedding with TSNE
embeddings = TSNE(verbose=2).fit_transform(train_sample)


# In[ ]:


# Visualize TSNE embedding
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

plt.figure(figsize=(10,7))
plt.scatter(vis_x, vis_y, c=label_sample, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()


# # Classification Models

# In[ ]:


import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X_train_flat = train_images.reshape(60000, -1)
X_test_flat = test_images.reshape(10000,-1)

# Keras inputs
x_train = np.expand_dims(train_images, axis=-1)
x_test = np.expand_dims(test_images, axis=-1)
y_train = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)


# ### k-NN Baselines
# Warning: This is very slow! Skip this if you want to directly try out the CNN model.

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
neigh.fit(X_train_flat, train_labels)
print("k-NN Test Accuracy:", neigh.score(X_test_flat, test_labels))


# This is very close to the reported accuracy of 91.56% by Clanuwat et al.

# ### Simple CNN with Keras

# In[ ]:


img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 12


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Again, the accuracy is very close to the 95.12% reported on the paper.

# ## References
# 
# 1. Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718
# 
# ### Links
# * Original Paper repository: https://github.com/rois-codh/kmnist
# * Multicore TSNE: https://github.com/DmitryUlyanov/Multicore-TSNE
# * Inspiration: https://www.tensorflow.org/tutorials/keras/basic_classification
