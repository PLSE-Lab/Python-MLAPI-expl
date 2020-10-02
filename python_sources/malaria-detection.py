#!/usr/bin/env python
# coding: utf-8

# # Malaria Detection using Keras
# 
# ![Malaria](http://www.ox.ac.uk/sites/files/oxford/styles/ow_medium_feature/public/field/field_image_main/Malaria%20banner_0.jpg?itok=d_zOvyQy)
# 
# In this tutorial, we are going to explore how to train our model to detect malaria parasite images from the test slides containing cells.
# For creating the model, we are going to use a very basic and popular Deep learning framework called as ***Keras*** that has been designed by one of Google's researcher.
# 
# This tutorial is going to guide you through building an end to end pipeline from reading images from the folder to making your predictions

# Here I have imported all basic Machine Learning libraries that we have already seen in the module 2 of the course. But apart from that the only other libraires I have imported are *Keras* and *OpenCV*.
# 
# 
# * *Keras* is the Deep learning library that is designed by a Google Researcher for ease of building high level models.
# * *CV2 (OpenCV)* is the computer vision library which we are going to use to read and write the images from within the folder.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


import cv2
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Input, BatchNormalization, ZeroPadding2D, Dropout
from keras.models import Sequential, Model


# In[ ]:


base = '../input/cell_images/cell_images/'
para = os.listdir(base+'Parasitized')
nor = os.listdir(base+'Uninfected')


# In[ ]:


len(para), len(nor)


# The *para* variable contains the images of the parasite slide whereas the *nor* dataset contains uninfected images. 

# In[ ]:


def image_reader(path):
    '''
    Image_reader:
        Takes image as input and returns the RGB image of the same read image
        The image is also resized into smaller shape and Dimension of 110x110.
    '''
    t=cv2.imread(path)
    t=cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    t=cv2.resize(t, (110,110))
    return t


# In[ ]:


X = []
Y = [] 

for x in para:
    # Parasite variable 
    try:
        t = image_reader(base+'Parasitized/'+x)
        X.append(t)
        Y.append(1)
    except:
        pass
    
for x in nor:
    # Non Parasite images variable
    try:
        t = image_reader(base+'Uninfected/'+x)
        X.append(t)
        Y.append(0)
    except:
        pass


# 1. The labels are numeric and needs to be converted to One-Hot encoded. If you are wondering what One-Hot-Encoding is, you should refer to this [Machine Learning Mastery article](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/). 

# In[ ]:


X=np.array(X)
Y=np.array(Y)
Y_oh = keras.utils.to_categorical(Y, num_classes=2)
print(X.shape, Y_oh.shape)


# In[ ]:


# Show training images
np.random.seed(10)

concat_img = None
for i in range(10):
    idx = np.random.randint(X.shape[0])
    if concat_img is None:
        concat_img = X[idx]
    else:
        concat_img = np.concatenate([concat_img, X[idx]], axis=1)
plt.figure(figsize=(15, 5)) 
plt.imshow(concat_img)


# Keeping in mind this Deep learning Life-Cycle, we are going to split the dataset into train, test and validation dataset. If you are wondering what these are, then refer to [his blog](https://machinelearningmastery.com/difference-test-validation-datasets/) right now.
# 
# We are splitting the dataset into the test-set and train-set just first just to evaluate our model after finalizing all the required Hyperparameters. At training time, we are going to call the validation split which is going to be a part of the training-set itself.

# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(X,Y_oh,test_size=0.1, shuffle=True)


# ## Understanding our model
# 
# The model we have created is for understanding of all the concepts one would require for understanding and building your deep learning models. The model we have created here uses functional model structure defined in Keras. 
# 
# We have carried out the following in the model-
# 
# 1. Take image input in shape of (110,110,3) shaped array
# 2. Perform a zero padding over the image spanning 3 pixels
# 3. Perform a 2D convolution with kernel shape of (3,3) and stride=1
# 4. Apply relu activation over the outcome
# 5. Perform MaxPooling operation over the image
# 6. Repeat 3-5 again and BatchNormalize
# 7. Convert the 2D image to linear array
# 8. Perform Dropout on half of the nodes
# 9. Create a Dense layer with 2 outputs

# In[ ]:


X_input = Input((110,110,3))

# Zero-Padding: pads the border of X_input with zeroes
X = ZeroPadding2D((3, 3))(X_input)

# CONV -> BN -> RELU Block applied to X
X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)

# MORE CONVS
X = MaxPooling2D((2, 2))(X)
#shortcut = X
X = Conv2D(32, (3, 3), strides = (1, 1), padding="same")(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)
X = Conv2D(32, (3, 3), strides = (1, 1), padding="same")(X)
X = BatchNormalization()(X)
#X = layers.add([X, shortcut])
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool')(X)

# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
X = Flatten()(X)

# MORE DENSE
X = Dense(128)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)
X = Dropout(0.5)(X)

X = Dense(2, activation='softmax', name='fc')(X)

# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
model = Model(inputs = X_input, outputs = X, name='HappyModel')


# The model has been coded and needs to be compiled by defining the optimizer to be used, The targetted loss, and the Metrics on which to check the outcomes. These 2 things are required to compile the model. Once compiled, we are going to run 5 epochs (num of iterations ) to fit the data into the model.

# In[ ]:


model.compile('SGD', 'categorical_crossentropy', ['acc'])


# In[ ]:


history = model.fit(train_x, train_y, validation_split=0.1, epochs=5, batch_size=128)


# Remember the test dataset we had split? We are going to test our model on the same dataset now just to see weather the model has been properly trained or not.

# In[ ]:


model.evaluate(test_x, test_y)


# We can see that the model gets around 95 percent accuracy. This is sufficient to understand the concepts, but when it comes to building a industry standard model, this is not going to fair well. What could be done for that? Well I leave it upto you to come up with some innovative ideas and discuss them in the discussion section.

# You can see the model functionality merely changing the idx on a desired number. 

# In[ ]:


idx=124
t = test_x[idx].reshape(1,110, 110, 3)
print(np.argmax(test_y[idx]), np.argmax(model.predict(t)))
plt.imshow(t[0])


# In[ ]:


model.save_weights('happy_weights.h5')

