#!/usr/bin/env python
# coding: utf-8

# # Convolution Neural Network in Keras
# This kernel is created form scratch and will show how to build CNN.
# 

# ### Content
# * Data Preprocessing
# * Data Spliting
# * Building CNN classifier
# * Training Classiifer
# * Visualization
# * Future Work
# 

# ### Importing Required Module

# In[ ]:


import cv2                                         # working with, mainly resizing, images
import numpy as np                                 # dealing with arrays
import os                                          # dealing with directories
from random import shuffle                         # mixing up or currently ordered data that might lead our network astray in training.
from keras.models import Sequential                # creating sequential model of CNN
from keras.layers import Convolution2D             # creating convolution layer
from keras.layers import MaxPooling2D              # creating maxpool layer
from keras.layers import Flatten                   # creating input vector for dense layer
from keras.layers import Dense                     # create dense layer or fully connected layer
from keras.layers import Dropout                   # use to avoid overfitting by droping some parameters
from keras.preprocessing import image              # generate image
import matplotlib.pyplot as plt                    # use for visualization
import warnings#
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# Defining training and testing directory.
# Defining Image Size.

# In[ ]:


TRAIN_DIR = '../input/training_set/training_set'
TEST_DIR = '../input/test_set/test_set'
IMG_SIZE = 64,64


# Creating list to store image name,labels,matrix(pixel value)

# In[ ]:


image_names = []
data_labels = []
data_images = []


# Defining Funtion for creating data which takes data from both test and training test.

# In[ ]:


def  create_data(DIR):
     for folder in os.listdir(TRAIN_DIR):
        for file in os.listdir(os.path.join(TRAIN_DIR,folder)):
            if file.endswith("jpg"):
                image_names.append(os.path.join(TRAIN_DIR,folder,file))
                data_labels.append(folder)
                img = cv2.imread(os.path.join(TRAIN_DIR,folder,file))
                im = cv2.resize(img,IMG_SIZE)
                data_images.append(im)
            else:
                continue


# In[ ]:


#calling functions to create data
create_data(TRAIN_DIR)
create_data(TEST_DIR)


# In[ ]:


data = np.array(data_images)


# In[ ]:


len(data_images)


# In[ ]:


data.shape


# Converting string label into 0/1 using LabelEncoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

le = LabelEncoder()
label = le.fit_transform(data_labels)


# ## Data Spliting

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(data,label,test_size=0.20,random_state=42)

print("X_train shape",X_train.shape)
print("X_test shape",X_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)


# ## Creating CNN model

# ### Basics of CNN
# 
# * **CNN**-A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
# 
# * **Convolution Layer**-The aim of convolution operation is to reduce the size of an image, by using feature detectors that keep only the specific patterns within the image. Stride is the number of pixels with which we slide the detector. If it is one, we are moving it one pixel each time and recording the value (adding up all the multiplied values). Many feature detectors are used, and the algorithm finds out what is the optimal way to filter images. 3 x 3 feature detector is commonly used, but other sizes can be used.
# 
# * **Max pooling Layer**-A pooling layer is another building block of a CNN. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network. Pooling layer operates on each feature map independently.
# 
# * **Droput**- It is use to avoid overfitting by droping some random parameters form layer.
# 
# * **Dense**- Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and downsampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.
# 
# 
# 
# **Architecture**
# * Creating a convolution network of alternate convolution and max pooling network.
# * Using dropout to avoid overfitting.
# * Two dense layer of 128 and 1 neuron.

# In[ ]:


classifier=Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(output_dim= 128, activation='relu'))
classifier.add(Dense(output_dim= 1, activation='sigmoid'))
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
classifier.summary()


# ImageDataGenerator use for generating batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
# 
# 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)


# ## Training Classifier

# In[ ]:


batch_size = 32
steps_per_epoch=len(X_train)
validation_steps=len(y_val)

history=classifier.fit_generator(
    train_datagen.flow(X_train,y_train, batch_size=batch_size),
    steps_per_epoch = steps_per_epoch,
    epochs = 4,
    verbose = 2,
    validation_data = (X_val,y_val),
    validation_steps = validation_steps)


# In[ ]:


classifier.save_weights('model.h5')


# ## Visualization

# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

pred = classifier.predict_classes(X_val)
cm = confusion_matrix(y_val,pred)

f,ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Purples",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# ## Accuracy and Prediction

# In[ ]:


accuracy=(cm[0][0]+cm[1][1])/len(y_val)
print(accuracy)


# In[ ]:


import numpy as np
from keras.preprocessing import image


test_image=image.load_img('../input/test_set/test_set/dogs/dog.4042.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict_classes(test_image)

if result[0][0] >=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)


# ## Future Work 
# * Improving Architecture
# * Increasing Data Size
# * Using hyper-parameter
# * Using Pretrained model such VGG

# #### you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated :)
# 

# In[ ]:




