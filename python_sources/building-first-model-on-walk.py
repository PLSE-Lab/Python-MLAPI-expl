#!/usr/bin/env python
# coding: utf-8

# **Building First Convoulutional Network: **
# 
# In this tutorial we will discuss the following:
# 1. Reading and preprocessing images from the directory. 
# 2. Builiding a simple CNN. 
# 3. Train Model
# 

# In[1]:


## import matplot ibrary of visulazing images and results
import matplotlib.pyplot as plt


# In[2]:


## import Kereas and its module for image processing and model building
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization


# **Data Visualization:**
# 
# Before diving deep into the work, we first exmained the data and its defined label. The data is captured in a walking simulator in building to go outside of rooms without colloiding objects/walls as first person to mimic the human walk, we labelled each view/image as right, left or forward when there is need to turn right/left or move forward respectively.
# 
# In the following code snippet the 'The images of three labels i.e. Forward, left and right are shown.

# In[3]:


img = load_img('../input/train/train/Forward/screen_640x480_2018-01-15_12-18-58.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)
print('image shape: ', x.shape)

print('Step Forward after recognizing viwe')
plt.imshow(img)
plt.show()


img = load_img('../input/train/train/Left/screen_640x480_2018-01-15_12-18-50.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)
print('Turn Left after recognizing viwe')
plt.imshow(img)
plt.show()

img = load_img('../input/train/train/Right/screen_640x480_2018-01-15_14-27-30.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)
print('Turn Right after recognizing viwe')
plt.imshow(img)
plt.show()


# In the first image of above three, we can easily see that there is enough room to walk ahead with less risk of colloiding with wall. 
# 
# And following two images shows that we must take a turn to avoid colloision.

# **Convolutional Neural Network: **
# 
# Convolutional Neural Networks similar to Aritifical Neural Netowrk in architecture in the sense that they both have input, hidden and ouput layers. Where input layers take data and output layer predict the label of class but the difference  is ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network. The following figures shows the architecture of both netowrk
# 
# ![Artificial Neural Network](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)
# >                                    Artificial Neural Network
# 
# ![Convolutional Neural Network](http://cs231n.github.io/assets/cnn/cnn.jpeg)
# >                                   Convolutional Neural Network
# 
# The typical architecture of Conv net contains three main layers  *Convolutional Layer*, *Pooling Layer*, and *Fully-Connected Layer*. The Starting layers of conv net consits of Conv and Pooling layers whereas the deep layers near to output layers are dense layers(fully connected layers).
# [Details of ConvNet](http://cs231n.github.io/convolutional-networks/)
# 
# Considering these layers we built simple Convolutional Neural Network as below, to make training computionally easier we also normalized the data between each layer with *model.add(BatchNormalization())* :
# 
# 

# In[4]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# The above model architecture will be of same shape like below:
# ![](https://cdn-images-1.medium.com/max/2000/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)

# In[5]:


# this is the augmentation configuration we will use for training
# The image values are rescaled to 0-1.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


#Number of Batches during each epoch
batch_size = 64


# In[7]:


# this is a generator that will read pictures found in
# subfolers of '../input/train/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../input/train/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../input/test/test',
        target_size=(150, 150),
        batch_size=batch_size)


# In[8]:


# summarize history for accuracy
def plot_res():
        # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[9]:


history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)


# In[10]:


# Plot accuracy and loss over each epoch
plot_res()


# ---
# **Prediction: **
# 
# In order to see how the model is predicting the action after recognizing images, we have get prediction on the same three images shown on the above section of the this notebook.

# In[11]:


import scipy
import numpy as np


# In[12]:


def get_reshape(x=img):
    img_resize = scipy.misc.imresize(x,(150,150,3))
    img_reshape = np.empty((1,150,150,3), dtype='float')
    img_reshape[0,:,:,:] = img_resize
#     print(img_reshape.shape)
    return(img_reshape)


# In[13]:


def get_prediction(img_reshaped):
    pred = history.model.predict_classes(img_reshaped)[0]
    
    
    for action, val in train_generator.class_indices.items():   
        if val == pred:
#             print(action)
            pred = action
    return(pred)


# In[14]:


img = load_img('../input/train/train/Forward/screen_640x480_2018-01-15_12-18-58.png')
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)

print('The label of image is marked as Forward')
plt.imshow(img)
plt.show()

img_reshaped = get_reshape(x)
pred = get_prediction(img_reshaped)
print('The predicted action to take is : ', pred)
print('-------------------------------------------')


# In[15]:


img = load_img('../input/train/train/Left/screen_640x480_2018-01-15_12-18-50.png')
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)

print('The label of image is marked as Left')
plt.imshow(img)
plt.show()

img_reshaped = get_reshape(x)
pred = get_prediction(img_reshaped)
print('The predicted action to take is : ', pred)
print('-------------------------------------------')


# In[16]:


img = load_img('../input/train/train/Right/screen_640x480_2018-01-15_14-27-30.png')
x = img_to_array(img)  # this is a Numpy array with shape (3, 480, 640)

print('The label of image is marked as Right')
plt.imshow(img)
plt.show()

img_reshaped = get_reshape(x)
pred = get_prediction(img_reshaped)
print('The predicted action to take is : ', pred)
print('-------------------------------------------')

