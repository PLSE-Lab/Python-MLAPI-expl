#!/usr/bin/env python
# coding: utf-8

# This notebook is intended to help beginners get started with image classification using the Keras framework. Keras allows you to create a neural network with only a few lines of code. (Keras is built on top of lower-level deep learning libraries. By default, it uses TensorFlow as the backend.) 
# 
# The notebook is split into three sections:
# 1. Preprocessing the data
# 2. Training the data, and 
# 3. Making predictions. 
# 
# Note: If you're interested in using TensorFlow directly, the [Beginner's Guide to Classification (TensorFlow)](https://www.kaggle.com/ndalziel/beginner-s-guide-to-classification-tensorflow/) will help you get started.

# ## 1. Preprocessing the data

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense


# Let's begin by reading in the training data, and taking a look at some sample images. Note that the data has already been partially pre-processed - the 28x28 data matrix for each image has been flattened into a 784-column vector. So, to display the images, we need to reshape into a 28x28 matrix. 

# In[ ]:


train = pd.read_csv('../input/train.csv')
X = train.loc[:,'pixel0':'pixel783']
Y = train.loc[:,'label']
X_test = pd.read_csv('../input/test.csv')

for n in range(1,10):
    plt.subplot(1,10,n)
    plt.imshow(X.iloc[n].values.reshape((28,28)),cmap='gray')
    plt.title(Y.iloc[n])


# Next, we'll split the data into a training set and a cross-validation set. We'll use dummy variables to encode the labels - resulting in a 10-column label matrix (one for each digit).  

# In[ ]:


# Create training data set
X_train = X[:40000]
Y_train = Y[:40000]
Y_train = pd.get_dummies(Y_train)

# Create cross-validation set
X_dev = X[40000:42000]
Y_dev = Y[40000:42000]
Y_dev = pd.get_dummies(Y_dev)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of cross-validation examples = " + str(X_dev.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("Y_dev shape: " + str(Y_dev.shape))


# ## 2. Training the model

# In Keras, you build the model one layer at a time, starting with the 2nd layer. For each layer, you need to specify the **number of units** (or nodes) and the **activation function**. For the 2nd layer, you also need to specify the number of inputs. (Note that also you have the option of adding regularization functions to each layer to reduce over-fitting.)
# 
# The **number of units** in the input layer has to correspond to the number  of features in the input vector, and the number of layers in the final layer has to correspond to the number of categories. So, we need 784 inputs (28x28) and an output for each of the 10 digits. In the model below, we have a 4-layer network, with 32 and 16 units in the 2nd and 3rd layers respectively.
# 
# There are several options for the **activation** function. However, as a starting point, the 'relu' function is the best option for the hidden layers. For the final output layer, you should use 'sigmoid' for binary classification, and 'softmax' for multi-class classification.

# In[ ]:


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Next, we'll specifiy how the model is optimized by choosing the **optimization algorithm** and the **cost (or loss) function.** The Adam optimization algorithm works well across a wide range of neural network architectures. (Adam essentially combined two other successful algorithms - gradient descent with momentum, and RMSProp.) For the loss function, 'binary_crossentropy' is a good choice for binary classification, and 'categorical_crossentropy' is a good choice for multi-class classification. 

# In[ ]:


model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])


# Finally, we're ready to run the model. We need to decide on the number **epochs**. or iterations,  and the **batch size**, which represents the number of training examples that will be processed in each epoch. (Using mini-batches accelerates convergence of the algorithm.)

# In[ ]:


model.fit(X_train.values, Y_train.values, epochs=20, batch_size=64,verbose=2,
          validation_data=(X_dev.values, Y_dev.values))


# ## 3. Making predictions
# Now, let's create the test set predictions and the submission file...

# In[ ]:


predictions = model.predict_classes(X_test.values, verbose=0)
predictions_df = pd.DataFrame (predictions,columns = ['Label'])
predictions_df['ImageID'] = predictions_df.index + 1
submission_df = predictions_df[predictions_df.columns[::-1]]
submission_df.to_csv("submission.csv", index=False, header=True)
submission_df.head()

