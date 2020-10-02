#!/usr/bin/env python
# coding: utf-8

# **Digit Classification on MNIST Dataset using Keras**

# In this example, I've trained a Convolution Neural Network from scratch on the MNIST Dataset
# This notebook is inspired by the book **Deep Learning with Python** written by Keras author Francois Chollet.
# 
# We'll be using Keras, a simple-to-use deep learning library in Python. Let us start by importing the required libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import  train_test_split
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


# Now when that's underway, let us **import the data**.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# We've created two dataframes, *train* and *test* to store the training and testing data respectively. Now let's extract the features and labels out of the training data. 
# 
# *Note: the 'test' dataframe is left unchanged here.* 

# In[ ]:


y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)
del train


# Thus, *X_train* and *y_train* contain training data and labels respectively, and *test* contains test data.

# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


test.shape


# **Preprocessing the data**
# 
# At present, each data value contains a pixel of range 0 to 255. We normalize them by so that each pixel value is between 0 and 1.
# 
# Next, we reshape them so that the linear array of (Size, 784) can be made to a shape of (Size, 28, 28, 1). This is the shape that will be required by our ConvNet. The general form is (Size, height, width, no. of channels).
# 
# *Note: This operation is applied to both training and testing data.*

# In[ ]:


X_train = X_train/255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test/255.0
test = test.values.reshape(-1, 28, 28, 1)


# Now let's come to the labels. First, take a look at them before we do anything.

# In[ ]:


y_train[9]


# In[ ]:


g = sns.countplot(y_train)


# Presently *y_train* contains the actual digit (from 0 through 9). We need to transform it into a more computer friendly form. We do this by **One-Hot Encoding**.

# In[ ]:


y_train = to_categorical(y_train, num_classes = 10)


# In[ ]:


y_train[9]


# See, *y_train[9]* now stores [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] instead of just 3. That is, an array of 10, where all are 0 expect the label, which is 1.
# 
# Now, let's crave the validaton set out of out training data. This is the data that our model evaluates on after each cycle of training. We use the Scikit Lean function ***train_test_split*** for this.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)


# **Building the Model. **
# 
# Our model consists of 3 Pairs of Conv2D-MaxPool layers and 2 Dense layers. Dropout is also used to fight overfitting.

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))


# This is what the model actually looks like.

# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])


# Since this is a multiclass classification problem, we use loss as *categorial_crossentropy*.
# 
# Now, **training the model**.

# In[ ]:


history = model.fit(X_train, y_train, epochs = 20, batch_size = 128, validation_data = (X_val, y_val), verbose = 2)


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']


# **Plotting** the training vs validation loss and accuracy, and checking how our model performed.

# In[ ]:


epochs = range(1, 21)

plt.plot(epochs, loss, 'ko', label = 'Training Loss')
plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


# In[ ]:


plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


# As you can see, our accuracy on validation set is above 99%. Let's see how well this model can do on test set.

# In[ ]:


results = model.predict(test)


# Presently, *results* contains One-Hot Vectors. We convert them back to digits by picking out the maximum index of the vector.

# In[ ]:


results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')


# Let's take a look at what our *results* look like.

# In[ ]:


results


# Makng a *submission.csv* file containing our results.

# In[ ]:


submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)
submission.to_csv("MNIST_Dataset_Submissions.csv", index = False)


# This is my first public kernel on Kaggle. Hope you enjoyed it. 
# 
# Any comments and suggestions are welcome :)
# 
# Checkout this repository for a slightly modified version of the above kernel: https://github.com/raahatg21/Digit-Recognition-MNIST-Dataset-with-Keras/blob/master/MNIST_9914.ipynb
# 
# Thanks for reading
# 
# *Raahat Gupta*
