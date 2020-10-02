#!/usr/bin/env python
# coding: utf-8

# #### First attempt at building a CNN using MNIST digit recognition data
# 
# Very grateful to other Kagglers, especially [Yassine Ghouzam](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) for posting their kernels that taught me some super neat tricks!
# 
# 1. Explanation: Jumanji
# 2. Preprocessing
#   - Load libraries
#   - Load and check data
#   - Reshape data into 2D images
#   - Normalize data
#   - Generate augmented images
#   - One-hot label encoding
#   - Training/validation split
# 3. Tonight on CNN
#   - Define and compile model
#   - Define learning rate callback
# 4. Evaluate CNN model
#   - Confusion matrix
# 5. Submit predictions

# ## 1. Jumanji
# 
# You've probably seen the movie Jumanji, or at least heard of it. I just learned there's a remake in theaters right now!!
# 
# <img src = "http://cloud-3.steamusercontent.com/ugc/91602828075643624/F16C1BD022982E1C587B0DBEF8B85D897C7B0DC9/">
# 
# But did you play the board game released shortly thereafter? Each turn, players draw cards, but they're essentially unreadable. Here's an example.
# 
# <img src="https://cf.geekdo-images.com/images/pic340327_md.jpg", alt="Jumanji Card", style="width: 250px;">
# 
# To read each card, one has to put it under a red piece of cellophane, after which one can clearly read the hidden blue text.
# 
# I'm going to try to use this idea to explain convolution neural networks, so bear with me! The situations are similar: we have some information embedded in a 2D image and we need to filter some parts of the image out to get to the information.
# 
# Imagine the Jumanji cards are more sophisticated: instead of solid red noise over the card, there are several overlapping gradients which obscure the text, and let's assume the situation is further complicated by the fact that we are colorblind and don't know what colors we'll need our filters to be! So we have several cards (digit images, in the case of MNIST), and there's some information we want to extract from each card, but we don't know what our filters should be in order to best decipher the information. If we can find the right filter(s), then they should work well for all the cards!
# 
# This is like our situation: to train a CNN, instead of sticking each entire image under one big filter, one looks for several *local* filters, since often clues about the information are local in the image. As another example, if we're trying to classify photos of cats and dogs, we'd like different properties of the fur, eye shape, etc., a chance at being their own features capable of predictive power.
# 
# It seems natural therefore to pick up clues by looking at different small areas of the card under different filters. This is what the first layer of a CNN does! It trains, via backpropagation, several local filters to decipher local information, i.e. local *features*. The collection of local filters makes up the first layer of our CNN, the convolution layer.
# 
# After passing the image through this layer, we pass it through an activation layer (we'll use a ReLU activation, that is, the function $x\to max(x,0)$) and a MaxPooling layer. The activation layer throws out sections of the filtered image which (at the current training stage) don't have good predictive power. The maxpool layer then picks out the strongest among neighboring local features; in our case, it simply picks out the greatest number in each 2x2 patch of features, therefore shrinking the 28x28 image to a 14x14 array of features
# 
# Sometimes, if there is fear of overfitting the neural network to some training data, there is also included a Dropout layer, which randomly drops a portion of the data, i.e. zeros the outputs of several nodes. For some reason I can't get this to work for me in Keras.
# 
# These groups of layers (Conv-Activation-MaxPooling-Dropout) are stacked, gradually merging local features into global features, until the image has been boiled down to a relatively small array of aspiring predictors.
# 
# Then, having distilled the image into features, we unravel the 2D array into a 1D array, and pass everything through a fully connected (Dense) layer so that arbitrary linear combinations of these features may have the chance to be good predictors. Without the fully connected layer, disparate sections of each image have only weak means of informing one another about the content of the overall image, that is, only at the final maxpooling stage when they are compared as neighbors.
# 
# And that's a crash course in convolutional neural networks, as I understand them! Please leave a comment if my intuition is wrong or incomplete, or if I've left out anything cool!

# ### 2. Preprocessing

# ##### Load libraries and set constants

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white')

SEED = 323
VAL_FRAC = 0.1
EPOCHS = 20
BATCH_SIZE = 128


# ##### Load data

# In[ ]:


if os.path.isfile('../input/train.csv') and os.path.isfile('../input/test.csv'):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    print('train.csv loaded: train({0[0]},{0[1]})'.format(train.shape))
    print('test.csv loaded: test({0[0]},{0[1]})'.format(test.shape))
else:
    print('Error: train.csv or test.csv not found in /input')
    
print('')
print(train.info())
print('')
print(train.isnull().any().describe())
print('')
print(train['label'].value_counts())


# Looks like the first column in our data contains the labels, i.e. the correct digits for each image. Let's split that column off and store it.

# In[ ]:


y_train = train['label']
X_train = train.iloc[:,1:]

print(y_train.shape, X_train.shape)
del train


# ##### Reshape samples into 2D images

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
X_test = test.values.reshape(-1,28,28,1)

print(X_train.shape, X_test.shape)


# ##### Normalize

# In[ ]:


X_train = X_train.astype(np.float) # convert from int64 to float32
X_test = X_test.astype(np.float)
X_train = np.multiply(X_train, 1.0 / 255.0)
X_test = np.multiply(X_test, 1.0 / 255.0)


# In[ ]:


#CHECK: plot some images
plt.figure(figsize=(18,2))
for i in range(12):
    plt.subplot(2,12,1+i)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(X_train[i].reshape(28,28),cmap=matplotlib.cm.binary)


# Interesting! I'm intrigued by the 1s in the first and third images. Seems to suggest we might make our algorithm more robust by generating new images from these via some subtle transformations. Keras has a method for this.

# ##### Generate augmented images

# In[ ]:


idg = ImageDataGenerator(rotation_range=8.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=np.pi/30., # 6 degrees
    zoom_range=0.1)

idg.fit(X_train)


# ##### One-hot encoding

# In[ ]:


y_train = to_categorical(y_train, num_classes = 10)


# ##### Training/validation split

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = VAL_FRAC)


# ## 3. Tonight on CNN

# Here's our model.
# 
# Input ->
# 
# Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPooling2D ->
# 
# Conv2D -> ReLU -> MaxPooling2D ->
# 
# Flatten -> Dense -> Relu -> BatchNormalization -> Dense -> Relu -> Dense -> Softmax -> Output
# 
# 
# For some reason, I couldn't get the Dropout layer to work for me in Keras (I'll be making another attempt soon now that I've uploaded this kernel to Kaggle!) so I had to pare down the complexity of my model so I didn't need to regularize it quite so much. Regularization is an umbrella term for any of a variety of techniques used when problems are ill-posed or models are prone to overfitting. In the case of neural networks, typically there are far more weights being trained than might be necessary to capture the relevant features, so the model fits to noise. Adding regularization keeps the model simpler and less prone to overfitting.

# ##### Define and compile model

# In[ ]:


model = Sequential()

model.add(Conv2D(16,(6,6), input_shape = (28,28,1), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(4,4), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3), activation = 'relu'))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer = Adam(),
             metrics = ['accuracy'])


# ##### Define learning rate callback

# In[ ]:


lr = ReduceLROnPlateau(monitor='val_loss',
                       factor=0.5,
                       patience=2,
                       verbose=1,
                       epsilon=0.0001,
                       min_lr=1e-6)


# In[ ]:


fit = model.fit_generator(idg.flow(X_train,y_train,
                          batch_size=BATCH_SIZE),
                          epochs=EPOCHS,
                          validation_data=(X_val,y_val),
                          verbose=2,
                          steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                          callbacks = [lr])


# ## 4. Evaluate CNN Model

# In[ ]:


y_pred = model.predict(X_val)
# Convert predictions to one-hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation set labels to one-hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
conf = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(conf,range(10),range(10)), annot=True)


# ## 5. Submit predictions

# In[ ]:


predictions = model.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)


# 
