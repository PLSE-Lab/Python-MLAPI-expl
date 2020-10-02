#!/usr/bin/env python
# coding: utf-8

# # Image Recognition with CNNs (keras)
# 
# This notebook serves as a quick demonstration of a simple and accurate keras CNN connected to a feed-forward neural network for classifying the MNIST dataset
# 
# First things first we need to import the needed libraries and load our data up!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Loading the data
# I'm adding the ".values" to the test set to convert it to a numpy array
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv").values

print(train.shape, test.shape)


# we need to split our training set into features and labels to facilitate our process, this is not needed in the test set as we can see that it has one less feature (the label)

# In[ ]:


# I'm adding the ".values" to convert them to numpy arrays for easier algebraic manipulation
train_y = train["label"].values
train_x = train.drop(labels = ["label"], axis = 1).values


# Conv2D layers (keras CNNs) assume our data comes in three dimensions: height, width and channel. Each picture in is 28x28 pixels (that's why we have 784 feature values) and we have just one channel value which is black and white
# 
# We need to reshape our datasets to match these required settings

# In[ ]:


train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test = test.reshape((test.shape[0], 28, 28, 1))


# Now that that's done we need to standerdize our data (which as of right now ranges from rgb values 0-254) so that every value ranges between 0 and 1

# In[ ]:


train_x = train_x.astype("float32") / 255.0
test = test.astype("float32") / 255.0


# Next we need to one-hot encode our labels. Our labels are simply the number the 784 pixels are representing (0 through 9) that means that in the last layer of our feed-forward neural network we will need 9 neurons to specify the probability of the number being 0, 1, 2 etc. Because in the training process of our network we need to compare this result the neural network gives us with the label, we need them to be in the exact same format in order to have a proper comparison, so we have to transform each number in our labels to an array of length 10 where we put a 1 in the index of the correct number, and zeros everywhere else. It's best to show by example:
# 
# 0 -> [**1**, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# <br/>
# 1 -> [0, **1**, 0, 0, 0, 0, 0, 0, 0, 0]
# <br/>
# 2 -> [0, 0, **1**, 0, 0, 0, 0, 0, 0, 0]
# <br/>
# 3 -> [0, 0, 0, **1**, 0, 0, 0, 0, 0, 0]
# <br/>
# ...

# In[ ]:


# We one-hot encode the trainning labels
train_y = keras.utils.to_categorical(train_y, 10)


# We'll also split our training data into 2, one for training another for validating

# In[ ]:


# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size = 0.1)


# And now we're ready to start building our model!
# 
# We'll define first the CNN part of our network, then well flatten it to conect to our fully conected feed-forward neural network

# In[ ]:


# We initialize the Keras model as a Sequential model
model = Sequential()

# The input layer is a convolutional layer with 32 filters
# The shape of the kernel in this layer is 3x3
# We add padding in this layer (so we can start the kernel right at the beginning of the image)
# and in this case we use padding "same" for it to add values to the padding that are copied from the original matrix (it could also be 0)
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))

# For this layer we add a ReLU activation
# We need to add ReLU because a convolution is still a linear transformation
# so we add ReLU for it to be a non linear transformation
model.add(Activation("relu"))

# We add batch normalization here
# This normalizes the output from the previous layer in order
# for the input of the next layer to be normalized
# In this case we put the channels at the end so we don't need to specify the axis of normalization
# otherwise we would need to specify
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding="same", activation = "relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())

# In this layer we Pool the layer before in order to reduce the number of features
# Since we are using a 2x2 pooling size we are keeping only half of the features in each dimension
# So instead of a 28*28 vector we now have a 14*14 tensor
# Since we are omitting the stride Keras assumes the same stride as pool size which is what we want
model.add(MaxPooling2D(pool_size=(2, 2)))

# We add a dropout layer of 25% dropout for regularization
model.add(Dropout(0.25))

# We add another convolution layer, in this case we don't need to specify the input shape
# because keras finds out the right input shape
model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))
model.add(BatchNormalization())

# After this pooling we have a 7*7 tensor
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Now we define a simple feed forward neural network

# In[ ]:


# We add a Flatten layer in order to transform the input tensor into a vector
# In this case we had a 7*7*64 (7*7*the number of filters we have)
model.add(Flatten())

# We have 512 neurons in this layer
model.add(Dense(512, activation = "relu"))

model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# Great! We have our neural network structured out! All we need now is to compile it, define a gradient descent optimizer and run it!
# 
# For the optimizer I choose just a the RMSProp gradient descent optimizer, but there's many other to choose from

# In[ ]:


# defining the learning rate, the number of epochs and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 30
BS = 86
opt = RMSprop(lr = INIT_LR, rho=0.9, epsilon=1e-08, decay=0.0)

# We track the metrics "accuracy"
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Reduce the learning rate by half if validation accuracy has not increased in the last 3 epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

fitted_network = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=BS, epochs=NUM_EPOCHS, callbacks=[learning_rate_reduction])


# Now that our network is trained, we just predict the training set and store everything the right way for submission!

# In[ ]:


# predict results
results = model.predict(test)

# now we want to retrieve the index that had the higher probability, that will be our prediction
results = np.argmax(results,axis = 1)

results = pd.Series(results, name = "Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index = False)

