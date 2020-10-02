#!/usr/bin/env python
# coding: utf-8

# 
# # About the data
# 
# The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. As noted in one recent replacement called the Fashion-MNIST dataset, the Zalando researchers quoted the startling claim that "Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel". To stimulate the community to develop more drop-in replacements, the Sign Language MNIST is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).
# 
# ## Load the dataset
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import Image
Image("../input/sign-language-mnist/amer_sign2.png")


# In[ ]:



train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')


# In[ ]:


train.shape


# 
# 
# The data set is given in the form of labels and pixel value ranging from pixel 1 to pixel 784 which is 28 * 28 image.
# 
# Let's see what does each sign means
# 

# In[ ]:


Image("../input/sign-language-mnist/american_sign_language.PNG")


# Each letter indicates a sign produced by our fingers. We will apply deep learning to these images to make sure our model can understand what sign indicated what letter.

# In[ ]:


labels = train['label'].values
#labels


# Every unique value finding and sorting

# In[ ]:


unique_value = np.array(labels)
np.unique(unique_value)


# # EDA

# In[ ]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# As you can see each one is almost equally distributed and droping the label coloumn from the training set

# In[ ]:


train.drop('label', axis = 1, inplace = True)


# Reshaping the images

# In[ ]:


images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# Since our target variable are in categorical(nomial) so we are using label binarizer

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels


# Lets see how the images look
# 

# In[ ]:


plt.imshow(images[0].reshape(28,28))


# Spliting the dataset into train(80%) and test(20%).

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 84)


# Lets using keras lib for deep learning

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


# Creating the batch size to 128 and using 50 epochs

# In[ ]:


batch_size = 128
num_classes = 24
epochs = 50


# Normalizing the training and test data

# In[ ]:


x_train = x_train / 255
x_test = x_test / 255


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# 
# 
# Visualizing the image after normalizing
# 

# In[ ]:


plt.imshow(x_train[0].reshape(28,28))


# # CNN Model

# In[ ]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))


# In[ ]:


model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# Fit and validate model

# In[ ]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy graph")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()


# As you can see, the number of epochs increase the accuracy also increases.
# 
# 
# 
# Now, let's validate with the test data
# 

# In[ ]:


test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# 
# 
# Predecting with test images
# 

# In[ ]:


y_pred = model.predict(test_images)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())


# # Conclusion
# 
# * If this tutorial is not enough you can check Deep Learning Tutorial for Beginners prepared by 
#     - https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
#     - https://www.kaggle.com/ranjeetjain3/deep-learning-using-sign-langugage
# * After this tutorial, my aim is to prepare 'kernel' which is connected to 'Novel Corona Virus' data.
# * If you have any suggestions, please could you write for me? I wil be happy for comment and critics!
# * Thank you for your suggestion and votes ;)
# 
# 

# In[ ]:




