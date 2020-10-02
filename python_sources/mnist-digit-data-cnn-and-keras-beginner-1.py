#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# I have some experience in classification neural nets from the 1990s (feed-forward backprops, adaptive resonance etc - all written in C or C++ from scratch) which took vectors as inputs. 
# I'm completely new to both Python (as a programming language) and the new net architectures (CNN etc) and frameworks (tensorflow, keras, sklearn) so armed with a couple of books and the on line documentation here's my first attempt at image classification using keras.
# 

# **Prepare the training data**
# 1. Read the csv file into a ndarray.
# 2. Separate the labels from the vectors of pixel data.
# 3. Normalize pixel values from {0,255} to {0,1}
# 4.  Reshape the input vectors from rows (elements) of 784 values to tensors 28x28 values with 1 channel (greyscale)
# 5. Recode the labels to a 10 element vector (corresponding to the 10 output neurons we will have in the output layer - one hot encoding) 
# 6. Separate (randomly) the training tensors into training and validation sets (9:1 split)
# 

# In[ ]:


# Imports for this block 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Load training data into an ndarray (skip the col headers - the first line in the csv file)
train = np.genfromtxt("../input/training/train.csv", skip_header=1, delimiter=",")

# Separate labels from data (for the train file): Labels are the first column
Y_train = train[:,0:1]
X_train = train[:,1:]
del train

# normalize integers (0-255) into floats (0-1)
X_train = X_train/255.0

# Reshape the ndarray from 42000,784,1 -> 42000,28,28,1 (the ,1 is the channel number - needed for conv nets)
X_train = X_train.reshape([-1,28,28,1])

# Encode labels to one hot vectors (eg : 2 -> [0,0,1,0,0,0,0,0,0,0] etc)
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the training set (using sklearn utility)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print('X_train shape = ', X_train.shape)
print('Y_train shape = ', Y_train.shape)
print('X_val shape = ', X_val.shape)
print('Y_val shape = ', Y_val.shape)


# **First Attempt  (VERY Naive)**
# 
# Very much a first attempt; simple convnet, no data augmentation, no dropout, no weight regularization in the dense layers etc.

# In[ ]:


# New imports for this block
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Build a quick conv net
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Let's see what the layers look like
print(model.summary())


# The summary above tells us a lot about what the net is going to do. It comprises 4 kinds of layers.
# 
# 1. Conv2D : this layer is a 'features finder'. It seeks translation invarient features in the input. Using the first layer as an example; it takes an input tensor (in the first layer this is a tensor of size 28x28x1) and passes a sliding window (3x3 in the first layer) over it, discovering features. The output is a new tensor of size 26x26x30. The number of filters (channels) 30 is indicitive of the fact that we have created 30 'views' of the input data showing the existence of particular features. Both the input and output tensors can be described as 'feature maps'. The X,Y size of the output feature map is smaller (28 -> 26) because not all (the outermost) pixels can be validly represented in the sliding windows. (See Keras docs for details {ref here}).
# 2. MaxPooling2D : this layer is an aggressive downscaling of the input tensor. It slides a 2x2 window over its input tensor with no overlap looking for max val. What we get is a new tensor with exactly half X,Y dimentions and the same number of  filters (channels). In our first MaxPooling2D layer we take a 26x26x30 input tensor and out put a new representation; a 13x13x30 tensor. It's all about taking what we've found so far and reducing its dimentionality.
# 3. Flatten : What we've got so far is OK but at some point we need to make a decision about things. The decision is going to be made by some feed-forward, back-propogation (of error) layers. These (Dense) layers take a 1D tensor (vector) as input and this is the job of the Flatten layer. In this case a 3x3x64 tensor is simply turned into a vector comprising 576 elements.
# 4. Dense : the 3 Dense layers take this 576 input vector and output a 10 element vector which can be used as input against an error function.
# 
# Let's just run the model, I'm setting GPU on! 30 epocs will run in under 3 mins on GPUs. On CPUs the same run would take much longer

# In[ ]:


# New imports for this block
from keras.optimizers import RMSprop

# Compile the model
model.compile(optimizer=RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

# Fit the model (running the validation test as we go and recording loss and accuracy on 'unseen' images)
history = model.fit(X_train, Y_train, epochs=30, batch_size=100, validation_data=(X_val, Y_val))


# OK:  about 99.47% accuracy in the validation set.
# Let's plot the trends on the  training and validation sets and see what our trends look like.

# In[ ]:


# New imports for this block
import matplotlib.pyplot as plt

# Get the loss and accuracy info from the history history dictionary
training_losses = history.history['loss']
validation_losses = history.history['val_loss']
training_acc = history.history['acc']
validation_acc = history.history['val_acc']

# label epochs from 1 not 0
epochs = range(1, 31)

plt.plot(epochs, training_losses, 'bo', label='Training loss')
plt.plot(epochs, validation_losses, 'b', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, training_acc, 'bo', label='Training accuracy')
plt.plot(epochs, validation_acc, 'b', label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# So, what we're not seeing is overfitting. Normally, we'd like to overfit and then generalize the net using dropout, layer removal or a bit of general tweaking. Additionally; if you look at the leader board for the MNIST competition you'll see that 0.994 won't get you into the top 50%. We need at least 3 9s in the validation set to be remotely competitive. If we wanted to save the net though, we'd do the folowing. 

# In[ ]:


# Save the net we have
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# and then use the model to make some predictions

# In[ ]:


# Load training data into an ndarray (skip the col headers - the first line in the csv file)
test = np.genfromtxt("../input/testing/test.csv", skip_header=1, delimiter=",")

# Normalize and reshape the data into 28x28x1 tensors
test = test/255.0
test = test.reshape([-1,28,28,1])

# Make a ndarray of digit probabilities
predictions = model.predict(test)

# Find the most probable digit in each row of the array
digit_predictions = np.argmax(predictions, axis=1)

# Write to file etc...

print('Done')


# That's it. A submission from this 'only' scores 0.968 accuracy on the 28000 test samples in the MNIST competition. However, it's a very first attempt and there are many modifications and optimisations that can be applied. A decent start I hope?

# 
