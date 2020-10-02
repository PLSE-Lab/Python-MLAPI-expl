#!/usr/bin/env python
# coding: utf-8

# **Summary**
# 
# 

# This is the 2nd part of my approach to build CNN network to classify xray images of chest to pneumonia class (1) or without signs of the disease. In the first part I used tensorflow library to build, train and test my model. In this approach I am going to use Keras library. I would like to show that implementation of the CNN in Keras is much more simpler (frankly speaking Keras is frontend to Tensorflow). The CNN architecture is the same as in the first approach. At the end I download ready to use CNN VGG16 to solve my classification problem basing on ready to use architecture. We will see the advatages of ready to use CNN.
# 

# **1. Data preparation**

# Assing paths to variables.

# In[ ]:


train_dir = "../input/chest_xray/chest_xray/train"
val_dir = "../input/chest_xray/chest_xray/val"
test_dir = "../input/chest_xray/chest_xray/test"


# Below I decide to use ready generator to rescale image pixel values into the 0-1 range. We make also some data augumentation on the training set. We only rescale our validation data to 0-1 range. 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')
    


# **2. Bulding CNN**

# I keep architecture of the CNN the same as I did in the first part. This is my CNN:
# 
# * Adam algorythm to optimize my loss function
# * Softmax cross entropy with logits as my cost function
# * input shape of images 150x150x1
# * First conv layer: 32 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2
# * Second conv layer: 64 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2
# * Third conv layer: 128 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2
# * Fourth conv layer: 128 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2
# * Flatting layer
# * Dense layer 512 neurons with relu activation
# * Dropout with 0.5
# * logits with 2 neurons and sigmoid as a activation

# In[ ]:


from keras import layers 
from keras import models


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()


# In[ ]:


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
,metrics=['acc'])


# **Training model**

# We train our network in 20 epochs. It will take some time so be patient. I traid to keep number of epochs and steps per epoch as short as I could due to computation time. If you have time be free to change parameters and you will see how this affect  CNN. 

# In[ ]:



history = model.fit_generator(train_generator,
steps_per_epoch=10,
epochs=20, validation_data=val_generator, validation_steps=20)


# As we can see we get good result on train set however acc values on validation set indicate that our CNN is overfiting. We could tune our hiperparameters however I would recommmend ready to use CNN architecture like VGG16 and then train our new CNN. 

# **VGG16**

# Why we use ready CNN? As you saw our dataset is very small one so our own CNN will be too weak to find appropraite weights to minimize our loss and make better prediction. Ofcourse we could make more data augumentation or make our CNNs more complicated to get better results. If we have enough power and time why not. But it is worth too mention that ready to use CNN are very usefull because a lot of computantional work was done by someone else like CNN tuning to get better results. I choosed VGG16 but there is much more traind CNN which can be used by you for example Resnet. 
# 
# VGG16 was trained on Imagenet dataset. This dataset is  big (15 milions images http://www.image-net.org/)  so if we know something about how CNN works we know that CNN at the begining is learning the structre of the image, characteristic shapes ect. Taking this into consideration we can get weights from the VGG16 which help our new CNN to learn better even on small dataset because we will give information to our new CNN how to extract characteristics from image. Thanks to that we will also save our time. If you want to know more about VGG16 please read this article: https://neurohive.io/en/popular-networks/vgg16/

# You have to know that we only use Convolution layers. We omit dense layer. Why we do that? Convolutions layers learn image representation so we can use weights from convolution layers to solve other problems. This is not possible in case of dense layers.

# In[ ]:


from keras.applications import VGG16


# In[ ]:


conv_basic = VGG16(weights='imagenet', include_top=False)


# We got weights from our VGG16. Argument include_top indicate that we do not use dense layer. Lets take a look on CNN architecture.

# In[ ]:


conv_basic.summary()


# We can approach to learing our new CNN in two ways. In the first approach we input our images to convolution layers to get numpy array with weights generated at the end of VGG16 Convolution layers. Than we take this weights and train on dense layers. This approach is simpler and faster because we do not have to train our network again on convolution layers which are computation time intensive. However we can not perform data augmentation in this scenario. The second approach assume that we train our network again on new architecture. In this scenario we can make some data augumentation however computation will take some time to learn new classificator. In my appraoch I will concentrate on the first scenario because of time. 

# In[ ]:


import numpy as np

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20 


# In this case we use again data generator but  only with rescaling to 0-1 range. Next we build function which extract features from our datasets. At the begining we create two empty numpy arrays filling them with zeros. Next we implement generator on our dataset to rescale our images to 0-1 range. Next we extract features using predict fucntion on our dataset images filling with them our numpys array. We extract also lables. 

# In[ ]:


def extract_features(directory, sample_count):
    
    features = np.zeros(shape=(sample_count, 4, 4, 512)) 
    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(
        directory, 
        target_size=(150, 150), 
        batch_size=batch_size, 
        class_mode='binary')
    
    i = 0
    
    for inputs_batch, labels_batch in generator:
        features_batch = conv_basic.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch 
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# In[ ]:


train_features, train_labels = extract_features(train_dir, 2000) 
val_features, validation_labels = extract_features(val_dir, 16) 
test_features, test_labels = extract_features(test_dir, 624)


# Our features shape will be (smaples,4,4,512). We have to flatten them to fit our dense connected clasificator.

# In[ ]:


train_features = np.reshape(train_features, (2000, 4 * 4 * 512)) 
validation_features = np.reshape(val_features, (16, 4 * 4 * 512)) 
test_features = np.reshape(test_features, (624, 4 * 4 * 512))


# Lets build and train our new model.

# In[ ]:


from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512)) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
              loss='binary_crossentropy',metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=30, batch_size=20, 
                    validation_data=(validation_features, validation_labels))


# A you can see above VGG16 with ready weights works much more faster and give better results. 
