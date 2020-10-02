#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# getting the labels corresponding to the image
label_df = pd.read_csv('/kaggle/input/crowd-counting/labels.csv')
label_df.columns = ['id' , 'people']
label_df.head()


# In[ ]:


# loading the images in vector format
img = np.load('/kaggle/input/crowd-counting/images.npy')
#img = img.reshape(img.shape[0], img.shape[1], img.shape[2], img.shape[3],1)
img.shape


# In[ ]:


labels = np.array(label_df['people'])
labels


# In[ ]:


# setting features and target value

x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.1)
print(x_train.shape[0])
print(x_test.shape[0])


# ***IMPORTANT - if you have a higher compute power then uncomment the cell below to normalize the values else dnt normalize since the notebook will crash due to memory error***

# In[ ]:


"""
x_train, x_test = x_train / 255.0, x_test / 255.0
"""


# In[ ]:


# create model 

model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(480,640,3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
    
])

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
model.summary()


# In[ ]:


# add a learning rate monitor to get the lr with smoothest prediction

lr_monitor = tf.keras.callbacks.LearningRateScheduler(
                lambda epochs : 1e-8 * 10 ** (epochs/20))


# In[ ]:


# train the model 

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, callbacks=[lr_monitor])


# In[ ]:


# plot mae
plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([np.min(history.history['lr']), np.max(history.history['lr']), np.min(history.history['loss']), 15])
plt.show()


# In[ ]:


np.max(history.history['lr'])


# ***seems like the model gives smooth results for (lr = 1e-6)***

# In[ ]:


# change the learning rate to 1e-5 and re-run the model

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=1e-6), metrics=['mae'])
model.summary()


# In[ ]:


# train the model 

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)


# In[ ]:


# plot mae
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.legend(['mae', 'val_mae'])
plt.ylim(1, 4)
plt.xlim(0, 50)

plt.xticks(np.arange(0,50, 5))

plt.xlabel('epochs')
plt.ylabel('mean absolute error')
plt.title('Mae in every epoch')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# set figure size

fig = plt.figure(figsize=(15,15))
grid = ImageGrid(
        fig, 111,
        nrows_ncols=(2,2),
        axes_pad=0.5
)

for x in range(0,4):
    
    grid[x].set_title('Number of people => ' + str(labels[x]))
    grid[x].imshow(img[x])
    


# ***for practical implementation set the threashold to 20 people or so depending on the area the camera covers***
# 
# **and if a certain threshold is reached then raise a red flag and clear the area**
# 
# **practice social distancing and be safe**

# In[ ]:




