#!/usr/bin/env python
# coding: utf-8

# # Sign Language

# In[ ]:


from IPython.display import Image
Image("../input/amer_sign2.png")


# # About the data

# The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. As noted in one recent replacement called the Fashion-MNIST dataset, the Zalando researchers quoted the startling claim that "Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel". To stimulate the community to develop more drop-in replacements, the Sign Language MNIST is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).

# Load the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/sign_mnist_train.csv')
test = pd.read_csv('../input/sign_mnist_test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# The data set is given in the form of labels and pixel value ranging from pixel 1 to pixel 784 which is 28 * 28 image.

# Let's see what does each sign means

# In[ ]:


Image("../input/american_sign_language.PNG")


# Each letter indicates a sign produced by our fingers. We will apply deep learning to these images to make sure our model can understand what sign indicated what letter

# In[ ]:


labels = train['label'].values


# In[ ]:


unique_val = np.array(labels)
np.unique(unique_val)


# # Data exploration

# In[ ]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# As you can see each one is almost equally distributed

# In[ ]:


train.drop('label', axis = 1, inplace = True)


# We are droping the label coloumn from the training set

# Re shaping the images

# In[ ]:


images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# Since our target variable are in categorical(nomial) so we are using label binarizer

# In[ ]:


images.shape


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


# In[ ]:


labels


# Lets see how the images look

# In[ ]:


plt.imshow(images[0].reshape(28,28))


# Spliting the dataset into train(70%) and test(30%)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


# In[ ]:





# For deep learning i am using keras library

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


# In[ ]:


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# Visualizing the image after normalizing

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


model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


# As you can see, the number of epochs increase the accuracy also increases.

# In[ ]:





# Let's validate with the test data

# In[ ]:


test_labels = test['label']


# In[ ]:


test.drop('label', axis = 1, inplace = True)


# In[ ]:


test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])


# In[ ]:


test_labels = label_binrizer.fit_transform(test_labels)


# In[ ]:


test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# In[ ]:


test_images.shape


# Predecting with test images

# In[ ]:


y_pred = model.predict(test_images)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(test_labels, y_pred.round())


# As we can see we got a really great accuracy 

# 

# We can increate the accuracy by tuning the hyper parameters of the model like playing with different activation functions and using different loss functions

# In[ ]:




