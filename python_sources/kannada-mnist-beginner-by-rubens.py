#!/usr/bin/env python
# coding: utf-8

#  - [Introduction](#section-one)
#  - [Data exploration](#section-two)
#  - [The model](#section-three)
#  - [What went wrong/right](#section-four)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a name="section-one"></a>
# ## Introduction
# 
# I am very much a begginer when it comes to data science, but I am really enjoying the learning process - so far anyway. I have been through many tutorials and I really need to start doing some competitions so I can actually learn. So here I am!
# 
# Often I find sarting the hardest part; this is especially true with deep learning I am finding. Classification and regression competitions seem much easier. 
# 
# So, here I am giving it a go in what I hope is a good starting point for a deep learning competition.
# 
# ### Progression
# 1. achieved 0.95120
# 2. added learning rate annealer and increased epochs by 10, achieved 0.97380
# 3. Changed from Sequential to functional - no change to score
#     - Decreased dropout slightly.
#     - Increased image zoom and height and width_shift_range from 0.1 to 0.25
# 4. added another dense layer and another convolutional layer, reduced the kernel size in the 1st layer. Also increased the test size by 10% - achieved 0.97620
# 5. Completely changed the model to this specification https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam. I still have no idea where people get the ideas for the design of these models, if you know please comment below. achieved 0.97780
# 6. changed optimizer from RMSprop to Adam 0.975
# 7. Edited Adam optimizer; increased epochs; changed learning rate to 0.33 from 0.5; reduced test size. 

# In[ ]:


#Import Packages

import random
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import cv2 as cv

print(cv.__version__)
print(np.__version__)


# <a name="section-two"></a>
# ## Data Exploration

# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
y_train = train['label']
X = train.drop(['label'],axis=1)
del train
Id = test['id'] # save this for the submission
test = test.drop(['id'],axis=1) #


# In[ ]:


X.shape


# In[ ]:


test.shape


# **Much fewer samples in the test set.

# In[ ]:


#plt.imshow(X.values[3].reshape(28,28))
plt.imshow(X.values[3].reshape(28,28), cmap = 'gray', interpolation = 'bicubic')


# ### Skeletonization function

# In[ ]:


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()
 
    skel[:,:] = 0
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    
    while True:
        eroded = cv.erode(img, kernel, iterations = 1)
#        eroded = cv.erode(img, kernel)
        temp = cv.dilate(eroded, kernel, iterations = 1)
        temp  = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)  
        
#        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
#        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
#        temp  = cv2.subtract(img, temp)
#        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv.countNonZero(img) == 0:
            break

    return skel


# In[ ]:


# runnig skeletonization in all images

# getting dimensions of the input images (X)
rows, columns = X.shape
#rows = 10

# configuring figure 
nrows = 10
ncols = 10
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))

# running all images
figureRow = 0
figureColumn = 0
for i in range(0, rows): # all images
    # reshaping image 
    xReshaped = X.values[i].reshape(28,28);
    
    # converting int64 to uint8
    xReshaped = np.uint8(xReshaped)
    
    # applyingthe  Otsu binarization
    threshold, xOtsuImage = cv.threshold(xReshaped, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # skeletonization image 
    xSkeletonized = skeletonize(xOtsuImage)

    # setting the image in the figure 
    if (i < (nrows * ncols)/2 ):
        ax[figureRow][figureColumn].axis('off')
        ax[figureRow][figureColumn].imshow(xReshaped, cmap = 'gray', interpolation = 'bicubic')
        ax[figureRow+1][figureColumn].axis('off')
        ax[figureRow+1][figureColumn].imshow(xSkeletonized, cmap = 'gray', interpolation = 'bicubic')

        # setting figures indexes
        figureColumn = figureColumn + 1
        if (figureColumn >= ncols):
            figureRow = figureRow + 2
            figureColumn = 0
        
    # setting X to skeletonized image 
    X.values[i] = xSkeletonized.flatten()
    
print(X.shape)


# In[ ]:


# runnig skeletonization in all images

# getting dimensions of the input images (X)
rows, columns = test.shape
#rows = 10

# configuring figure 
nrows = 10
ncols = 10
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))

# running all images
figureRow = 0
figureColumn = 0
for i in range(0, rows): # all images
    # reshaping image 
    testReshaped = test.values[i].reshape(28,28);
    
    # converting int64 to uint8
    testReshaped = np.uint8(testReshaped)
    
    # applyingthe  Otsu binarization
    threshold, testOtsuImage = cv.threshold(testReshaped, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # skeletonization image 
    testSkeletonized = skeletonize(testOtsuImage)

    # setting the image in the figure 
    if (i < (nrows * ncols)/2 ):
        ax[figureRow][figureColumn].axis('off')
        ax[figureRow][figureColumn].imshow(testReshaped, cmap = 'gray', interpolation = 'bicubic')
        ax[figureRow+1][figureColumn].axis('off')
        ax[figureRow+1][figureColumn].imshow(testSkeletonized, cmap = 'gray', interpolation = 'bicubic')

        # setting figures indexes
        figureColumn = figureColumn + 1
        if (figureColumn >= ncols):
            figureRow = figureRow + 2
            figureColumn = 0
        
    # setting X to skeletonized image 
    test.values[i] = testSkeletonized.flatten()
    
print(test.shape)


# Interesting, I cannot distinguish what this is, I hope a machine can.

# In[ ]:


# code for this plot taken from https://www.kaggle.com/josephvm/kannada-with-pytorch

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15,15))

# I know these for loops look weird, but this way num_i is only computed once for each class
for i in range(10): # Column by column
    num_i = X[y_train == i]
    ax[0][i].set_title(i)
    for j in range(10): # Row by row
        ax[j][i].axis('off')
        #ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28))
        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap = 'gray', interpolation = 'bicubic')


# In hindsight, if I were developing a number system, I would make sure that the symbols are as different as possible - 3 and 7 are basically the same, as are 6 and 9! Equations in kannada are probably quite interesting

# <a name="section-three"></a>
# ## Building the model

# In[ ]:


# re-shaping the data so that keras can use it, this is something that trips me up every time

X = X.values.reshape(X.shape[0], 28, 28,1)
X.shape
test = test.values.reshape(test.shape[0], 28, 28,1)
test.shape


# In[ ]:


# This modifies some images slightly, I have seen this in a few tutorials and it usually makes the model more accurate. As a beginner, it goes without saying I don't fully understand all the parameters

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.25, # Randomly zoom image 
        width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X)


# In[ ]:


y_train = to_categorical(y_train,num_classes=10) # the labels need to be one-hot encoded, this is something else I usually forget


# In[ ]:


# this model was taken from https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam

def build_model(input_shape=(28, 28, 1), classes = 10):
    
    activation = 'relu'
    padding = 'same'
    gamma_initializer = 'uniform'
    
    input_layer = Input(shape=input_shape)
    hidden=Conv2D(16, (3,3), strides=1, padding=padding, activation = activation, name="conv1")(input_layer)
    hidden=BatchNormalization(axis =1, momentum=0.1, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch1")(hidden)
    hidden=Dropout(0.1)(hidden)
    
    hidden=Conv2D(32, (3,3), strides=1, padding=padding,activation = activation, name="conv2")(hidden)
    hidden=BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch2")(hidden)
    hidden=Dropout(0.15)(hidden)
    hidden=MaxPool2D(pool_size=2, strides=2, padding=padding, name="max2")(hidden)
    
    hidden=Conv2D(64, (5,5), strides=1, padding =padding, activation = activation,  name="conv3")(hidden)
    hidden=BatchNormalization(axis =1,momentum=0.17, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch3")(hidden)
    hidden=MaxPool2D(pool_size=2, strides=2, padding="same", name="max3")(hidden)
    
    hidden=Conv2D(128, (5,5), strides=1, padding=padding, activation = activation, name="conv4")(hidden)
    hidden=BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch4")(hidden)
    hidden=Dropout(0.17)(hidden)
    
    hidden=Conv2D(64, (3,3), strides=1, padding=padding, activation = activation, name="conv5")(hidden)
    hidden=BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch5")(hidden)
    hidden=Dropout(0.2)(hidden)
    
    hidden=Conv2D(32, (3,3), strides=1, padding=padding, activation = activation, name="conv6")(hidden)
    hidden=BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch6" )(hidden)
    hidden=Dropout(0.05)(hidden)

    hidden=Flatten()(hidden)
    hidden=Dense(50,activation = activation, name="Dense1")(hidden)
    hidden=Dropout(0.05)(hidden)
    hidden=Dense(25, activation = activation, name="Dense2")(hidden)
    hidden=Dropout(0.03)(hidden)
    output = Dense(classes, activation = "softmax")(hidden)
    
    model = Model(inputs=input_layer, outputs=output)
    
    return model


# In[ ]:


# this model was taken from hans_on of the RNP4CV 

def build_model_rubens(input_shape=(28, 28, 1), classes = 10):

    model = Sequential()

    model.add(Conv2D(input_shape=input_shape, 
                     filters=10, 
                     kernel_size=(5, 5), 
                     strides=(1, 1), 
                     padding='valid', 
                     activation='relu', 
                     kernel_initializer='glorot_uniform'))

    model.add(MaxPool2D(pool_size=(2, 2), 
                           padding='valid'))

    model.add(Dropout(rate=0.3))

    model.add(Conv2D(filters=2, 
                     kernel_size=(10, 10), 
                     strides=(1, 1), 
                     padding='valid',
                     activation='relu',
                     kernel_initializer='glorot_uniform'))

    model.add(MaxPool2D(pool_size=(2, 2), 
                           padding='valid'))

    model.add(Dropout(rate=0.3))

    model.add(Flatten())

    model.add(Dense(units=4, 
                    activation='relu',
                    use_bias=True,
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros'))

    model.add(Dense(units=classes, 
                    activation='softmax',
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros'))

    model.summary()
    
    return model


# The main change to this model, other than the amount of layers, is batch normalization.
# 
# From what I have read and, let's face it, watched on YouTube, the reason why this is done is to keep the data on a similar scale and prevent gradient explosions. Apparently, non-normalised data can decrease training speed. It helps to stop the weights of the model becoming inbalanced?
# 
# here is a usefull if you're interested https://www.youtube.com/watch?v=dXB-KQYkzNU
# 
# One other thing I noticed about this model is that the dropout is smaller than I have seen in others. My guess is that becuase there are more layers the dropout is smaller in each layer, but overall it is similar? The combined number in this model is 0.75
# 
# 

# In[ ]:


# Define the optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model = build_model(input_shape=(28, 28, 1), classes = 10)
#model = build_model_rubens(input_shape=(28, 28, 1), classes = 10)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=10, 
                                            factor=0.33, 
                                            min_lr=0.00001)
# this could use some tuning, i don't think it needs to drop off so sharply.


# In[ ]:


epochs = 50 
batch_size = 100


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.1, random_state=1)


# In[ ]:


history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size ),
                              epochs = epochs,
                              validation_data = (X_val,y_val),
                              verbose = 1,
                              steps_per_epoch = X_train.shape[0] // batch_size,
                             callbacks = [learning_rate_reduction])

# On the first attempt I forgot to add the 'learning_rate_reduction'


# In[ ]:


model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# testing dropout
# 
# - 10 epochs o.g acc = 0.9765
# - all dropout at the end = 0.7684, but val accuracy was still 0.989
# - with 0.4 on each layer apart from the last 0.9685 and val of 0.9899
# - 0.1 on each layer with 0.4 on the two dense layers 0.9535 and val of 0.9903
# 
# in summary -- is there any point on the valdiation set?

# <a name="section-four"></a>
# ## How did we do?

# In[ ]:


fig, ax = plt.subplots(2,1)

ax[0].set_title('Loss function') # adding title to subplot
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best')

ax[1].set_title('Accuracy')  # adding title to subplot
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best')

fig.tight_layout()   # adjust space between subplots


# In[ ]:


import seaborn as sns

# used the code from https://www.kaggle.com/shahules/indian-way-to-learn-cnn to create this

y_pre_test=model.predict(X_val)
y_pre_test=np.argmax(y_pre_test,axis=1)
y_test=np.argmax(y_val,axis=1)

conf=confusion_matrix(y_test,y_pre_test)
conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))



# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(conf, annot=True)


# In[ ]:


print('out of {} samples, we got {} incorrect'.format(len(X_train), round(len(X_train) - history.history['accuracy'][-1] * len(X_train))))
print('accuracy {}',  history.history['accuracy'])


# In[ ]:


# code taken from https://stackoverflow.com/questions/55074681/how-to-find-the-wrong-predictions-in-keras - haven't yet figured out why my labels are onehot encoded still

y_pre_test=model.predict(X_val)
y_pre_test=np.argmax(y_pre_test,axis=1)
y_test=np.argmax(y_val,axis=1)

x=(y_pre_test-y_test!=0).tolist()
x=[i for i,l in enumerate(x) if l!=False]



fig,ax=plt.subplots(1,4,sharey=False,figsize=(15,15))

for i in range(4):
    ax[i].imshow(X_val[x[i]][:,:,0])
    ax[i].set_xlabel('Real {}, Predicted {}'.format(y_val[x[i]],y_pre_test[x[i]]))


# Now to make our predictions and submit!

# In[ ]:


predictions = model.predict(test)


# #### This made me stumble when I changed
# 
# With sequential there is a function to get the classes, but this is not true with the functional API. You get an array with the models probabilities for each class. Therefore you have to take the highest value from the array and get the corresponding class.

# In[ ]:


plt.imshow(test[0].reshape(28,28))


# In[ ]:


predictions[0] # looks like the 4 element is what the model thinks it is and it is over 99.99 % sure


# In[ ]:



predictions = predictions.argmax(axis = -1)
predictions


# In[ ]:


submission = pd.DataFrame({ 'id': Id,
                            'label': predictions })
submission.to_csv(path_or_buf ="Kannada_mnist.csv", index=False)

