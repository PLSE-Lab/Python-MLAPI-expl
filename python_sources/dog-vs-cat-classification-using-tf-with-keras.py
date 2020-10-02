#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing the modulus
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense  , Activation
from keras import applications
import re


# Visualizing a image of both cat and dog

# In[ ]:


import matplotlib.pyplot as plt
fig,axes=plt.subplots(nrows=1,ncols=2)

img1=load_img('../input/dogs-vs-cats-redux-kernels-edition/train/dog.1.jpg',target_size=(150,150))
axes[0].imshow(img1)
axes[0].set_title("Cat")
img2=load_img('../input/dogs-vs-cats-redux-kernels-edition/train/cat.1.jpg',target_size=(150,150))
axes[1].imshow(img2)
axes[1].set_title("Dog")


# In[ ]:


# Setting the parameters 
img_width = 150
img_height = 150
TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)] 


# In[ ]:


# images in directory path
train_images_dogs_cats[:5]


# ### Sorting the images in their directory by number in image name.
# Helper function for natural Sorting the images of dog and cat in their directory.                                                                                             
# In a classical alphanumerical sort we will have something like : 1 10 11 12 2 20 21 3 4 5 6 7                                                                           
# If you're using Natural ordering, it will be :1 2 3 4 5 6 7 10 11 12 20 21

# In[ ]:


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# In[ ]:


#Natural sorting on training data
train_images_dogs_cats.sort(key=natural_keys)


# In[ ]:


# After sorting directory
train_images_dogs_cats[:5]


# In[ ]:


# dog images starts from index 12500
train_images_dogs_cats[12500:12505]


# ## Chosing a subset of training images.
# The dataset consist of 25000 images of both cat and dog i.e 12500 images of each dog and cat.                                                               
# We will be using 3000 images of  cat and dog each . A total of which is 6000 images combined dog and cat.                                         
# Cat images ranges from index 0 to 12499 and dog images ranges from index 12500 to 25499
# So, selecting first 3000 images which are cat .
# Dog images start from index 12500 and goes to 15500 for 3000 images of dog.

# In[ ]:


train_images_dogs_cats = train_images_dogs_cats[0:3000] + train_images_dogs_cats[12500:15500] 


# In[ ]:


# natural sorting the test images
test_images_dogs_cats.sort(key=natural_keys)


# **Data Preprocessing** converting images to array

# In[ ]:


def prepare_data(list_of_images):
    """
    Returns a array of images
    
    """
    x = [] # images as arrays
    for image in list_of_images:
        x.append(img_to_array(load_img(image,target_size=(img_width,img_height))))
    return x


# Preparing data and generating their labels

# In[ ]:


X=prepare_data(train_images_dogs_cats)
X=np.array(X)
y=np.array([0]*3000 + [1]*3000)


# In[ ]:


# Spliting the data in training and validation data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1) 


# In[ ]:


# Shape of training data
print('Shape of training data {}'.format(X_train.shape))


# In[ ]:


nb_train=len(X_train)
nb_validation=len(X_val)
batch_size=16


# # Using the bottleneck features of a pre-trained network                                                                                                                       
# A more refined approach would be to leverage a network pre-trained on a large dataset. Such a network would have already learned features that are useful for most computer vision problems, and leveraging such features would allow us to reach a better accuracy than any method that would only rely on the available data.
# 
# We will use the VGG16 architecture, pre-trained on the ImageNet dataset --a model previously featured on this blog. Because the ImageNet dataset contains several "cat" classes (persian cat, siamese cat...) and many "dog" classes among its total of 1000 classes, this model will already have learned features that are relevant to our classification problem. In fact, it is possible that merely recording the softmax predictions of the model over our data rather than the bottleneck features would be enough to solve our dogs vs. cats classification problem extremely well. However, the method we present here is more likely to generalize well to a broader range of problems, including problems featuring classes absent from ImageNet.
# 
# Our strategy will be as follow: we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. We will then run this model on our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.                                                                                                                                                    Check VGG16 model here **
# [https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3](http://)

# In[ ]:


datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
model_vgg16 = applications.VGG16(include_top=False, weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
# creating a train generator 
generator = datagen.flow(X_train,y_train, batch_size=batch_size,shuffle=False)
    
# getting training data bottle neck features from  VGG16 model
bottleneck_features_train = model_vgg16.predict_generator(generator, nb_train // batch_size ,verbose=1 ) 


# In[ ]:


# validation generator
generator = datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=False)
    
# getting validation data bottle neck features from VGG16 model
bottleneck_features_validation = model_vgg16.predict_generator(
        generator, nb_validation // batch_size , verbose =1 ) 


# In[ ]:


# shape of bottleneck features
print("Shape of training bottleneck feature {}".format(bottleneck_features_train.shape))


# # Building the top model

# In[ ]:


model = Sequential()
#input layer
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))

# first hidden layer
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))

# Second hidden layer
model.add(Dense(units=126,activation='relu'))
model.add(Dropout(0.4))

#output layer
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

#fitting the model
history=model.fit(x=bottleneck_features_train,y=y_train,
                  epochs=10,
                  batch_size=batch_size,
                  validation_data=(bottleneck_features_validation,y_val))

model.save('TL_model.h5')


# In[ ]:


# visualizing the model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history['acc'],label='Train_acc')
plt.plot(history.history['val_acc'],label='val_acc')
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title('Model Accuracy Graph')
plt.legend()
plt.show()


# In[ ]:


# predicting result of test data
X_test = prepare_data(test_images_dogs_cats)
X_test=np.array(X_test)
print('Shape of test data {}'.format(X_test.shape))


# In[ ]:


nb_test=len(X_test)
batch_size=20


# # Predicting the result of test data                                                                                                                                                            
#  Here first we calculate the bottle neck features of our test data and give this features to input to top model to get the predictions.

# In[ ]:


# test data generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow( X_test, batch_size=batch_size,shuffle=False)

model_vgg16 = applications.VGG16(include_top=False, weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

#bottle neck features of our test data
bottleneck_feature_test_data=model_vgg16.predict_generator(test_generator,nb_test//batch_size,verbose=1)

# predicting probablities
prediction_probabilities = model.predict(bottleneck_feature_test_data,verbose=1) 


# In[ ]:


# Visualizing the result of prediction.
for i in range(10):
    if prediction_probabilities[i][0] > 0.5:
        print("I am {a:.2%} sure I am Dog".format(a=prediction_probabilities[i][0]))
    else:
        print("I am {a:.2%} sure I am Cat".format(a=(1-prediction_probabilities[i][0])))
    plt.imshow(load_img(test_images_dogs_cats[i],target_size=(150,150)))
    plt.show()


# In[ ]:


# creating a submission file
counter = range(1, len(test_images_dogs_cats) +1 )
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("dogsVScats.csv", index = False)

