#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This dataset is a fashion dataset, where thousands of small images are collected and labeled. Therefore it is a supervised problem. There are multiple ways to conquer this task, but my approach was to use a CNN to predict the fashion images.
# 
# There are two main files. The "fashion-mnist_train.csv" and the "fasion-mnist_test.csv" file. Both are csv files and they do have in the first column the label corresponding to the correct output. Then we have a lot of pixel values in each single other column.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# With those two lines we can see the data and how it is structured. We see that the first column is the label column, so the correct output, which are integers from 0 to 9. Then we have the pixel values in each other column. We see that these values are some weird rgb values. So we need to normalize this data and divide it by 255 to get the actually pixel value between 0 and 1.

# In[ ]:


df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_train.head()


# ## Seperating the data
# Now, we need to seperate the features and the labels. There are many ways you can seperate the data, but I've used iloc to seperate it.

# In[ ]:


labels = df_train.iloc[:,0]
labels.head()


# In[ ]:


features = df_train.iloc[:,1:]
features.head()


# Here we have a given array and a function which returns the actual class of the image. For example, we have a image showing a pullover. Therefore, we need to select 'class_names[2]' to get the corresponding name. We are doing this, because the labels are only numbers and we need a way to output everything for a actual human.

# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_name_of_label(label):
    return class_names[label]


# Now we are plotting some test data, to get a feeling how the data looks like. We are setting the corresponding label as the plot title.
# 
# As I said before we need to divide the values of this feature by 255 because we need to get the rgb value between 0 and 1. Then it is easier for the CNN to learn and to visualize it.

# In[ ]:


plt.figure(figsize=(12, 8))
for i in range(1, 6):
    plt.subplot(150+i)
    plt.imshow(np.reshape(features.loc[i].values/255, (28, 28)), 'gray')
    plt.title(get_name_of_label(labels[i]))
    plt.xticks([])
    plt.yticks([])


# ## Training the CNN
# 
# We are reshaping the values, because we just want a representation of the image in the size of 28x28.

# In[ ]:


features = features.values.reshape(-1, 28, 28, 1) 


# Now we are creating the model by using keras in tensorflow. We do have a input layer of 28x28. For representing every single pixel.
# 
# We do have a conv 2d layer of 32 neurons and with a scanning size of 3x3. We are using the relu activation function.
# 
# Then we have a max pooling of a 2x2 grid and again a layer of 64 convolutional neurons with the relu activation function. After that we have the pooling layer again.
# 
# Furthermore, we are flattening the whole thing and then we will connect 128 neurons with this flattened layer. We will use the relu activation function again.
# 
# In the end, we will use 10 output neurons (because we do have 10 output possibilities) with the softmax activation function to get a prober probability value for each output neuron.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# We will use the adam optimizer and the sparse categorical crossentropy loss function, because the adam optimizer is the best for this problem and the loss function is pretty default for a classical cnn.

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Here we are training the model by normalizing the features again. We will use hyperparameters epochs = 10 and batch size = 32. Furthermore, we will use a 80/20 split for validation.

# In[ ]:


history = model.fit(features/255, labels, epochs=3, batch_size=32, validation_split=0.2, shuffle=True)


# ### Plotting the model
# 
# In this section we will plot the accuracy and the loss of the model. This is good for seeing, if we have a overfit/underfit or a good model.

# In[ ]:


plt.figure(figsize=(18, 6))

plt.subplot(231)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.subplot(232)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()


# ## Testing the model
# 
# In this section we will test the model by using the 'fashion-mnist_test.csv' file.

# In[ ]:


df_test = pd.read_csv('../input/fashion-mnist_test.csv')
df_test.head()


# Here we are seperating the labels and the features again. The structure of the data is the same as in the training file.

# In[ ]:


test_labels = df_test['label']
test_features = df_test.iloc[:, 1:]


# Now we will evaluate the model by reshaping the test features values again to the corresponding 28x28 size.

# In[ ]:


test_loss, test_accuracy = model.evaluate(test_features.values.reshape(-1, 28, 28, 1), test_labels)


# ### Plotting the test data
# 
# In this small section we will use some test data and predict the output of this image. We will use the predict function to get the predictions of this image, that the model has made.

# In[ ]:


predictions = []
for i in range(0, 20):
    predictions.append(model.predict(test_features.loc[i].values.reshape(-1, 28, 28, 1)))


# In[ ]:


plt.figure(figsize=(12, 8))
for i in range(0, 20):
    plt.subplot(4, 5, i+1)
    plt.imshow(np.reshape(test_features.loc[i].values/255, (28, 28)), 'gray')
    plt.xticks([])
    plt.yticks([])
    prediction = np.argmax(predictions[i])
    plt.title('{} | {}'.format(get_name_of_label(test_labels[i]), get_name_of_label(prediction)))


# ## Conclusion
# 
# There are many ways to classify those images, but my personal favourite is the CNN way. CNN's are good for classification problems and therefore it is a good thing to test your knowledge with it.
# 
# Furthermore you can test this notebook and maybe built a better solution by optimizing the hyperparameters.
