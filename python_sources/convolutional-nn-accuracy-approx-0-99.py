#!/usr/bin/env python
# coding: utf-8

# ![](https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)

# Convolutional Neural Network is an Deep Learning algorithm, which takes the image as an input and specify the weights and biases to various objects/aspects in the image and that will be used to differentiate between images. The above image will summarize the process taken place in the CNN.
# 
# 1. The input image is sent into the CNN which will first convert it to the feature image.
# 2. The feature image will undergo Pooling process.
# 3. The pooled feature image data will be flatten into array
# 4. The flatten data will be used to train the NN
# 
# This kernel will undergo the following steps:
# 1. Loading the necessary libraries
# 2. Initialize the Neural Network
# 3. Adding Convolution and MaxPooling
# 4. Flattening
# 5. Creation of NN layers
# 6. Loading training and test set using Keras ImageDataGenerator
# 7. Training the model

# ### Note:
# **This is my first CNN, so if there is any improvements, please share it in the comments. I will learn and update my kernal. Thank you :)**

# #### 1. Loading the necessary libraries

# In[ ]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing import image


# #### 2. Initialize the Neural Network
# We will now initialize the neural network using `Sequential` and this will be used to train the image dataset

# In[ ]:


classifier = Sequential()


# #### 3. Adding Convolution and MaxPooling
# Now, we will be adding two layers in the NN model. 
# 1. Convolution2D - The input passed with this layer are: 
#     * Dense
#     * shape
#     * input_shape -> The input image shape will be of 64 x 64 size. Since the input image is a color-image, we will be passing 3 as the third parameter(R,G,B)
#     * activation -> it is the activation function (relu) that will be used in the convolutional layer.

# In[ ]:


classifier.add(Convolution2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))


# 2. Pooling - We will be adding the MaxPooling of pool size 2 x 2

# In[ ]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# Let's add another convolutional layer in order to improve the accuracy of the model. (With single convolutional layer, the accuracy was approx. 88%)

# In[ ]:


classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# #### 4. Flattening
# The final step of formatting the input data for training the NN is Flattening - converting the image data to a 1D format.

# In[ ]:


classifier.add(Flatten())


# #### 5. Creation of NN layers

# We will be adding two Neural layers of `relu` activation and `sigmoid` activation

# In[ ]:


classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))


# Binary crossentropy is used as a loss function since the classification is a binary (normal or pothole road)

# In[ ]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# #### 6. Loading training and test set using Keras ImageDataGenerator

# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('../input/pothole-images/Archive/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('../input/pothole-images/Archive/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# #### 7. Training the model

# When fitting the data in the model, we will pass the training set with 562 `steps_per_epoch` (since we have 562 training images) and pass the test set as `validation_data` with 101 `validation_steps` (we have 101 test images)

# In[ ]:


classifier.fit_generator(training_set,
        steps_per_epoch=562,
        epochs=25,
        validation_data=test_set,
        validation_steps=101)


# Now, the model has been trained by the input image datasets. We can plot the accuracy and the value accuracy of the classifier model

# In[ ]:


plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.title('Analysis of the model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Value Accuracy'], loc='upper right')
plt.show()


# Let's predict some images which is not seen by the model. The prediction images are loaded under the folder `pothole_image_prediction`. The type of the prediction images are:
# 1. Pothole road
# 2. Normal road
# 3. Normal road
# 4. Pothole road
# 5. Pothole road
# 
# Below is a function which takes the image path as input and predict the image and return the value.
# **Note: 0 - Normal and 1 - Pothole**

# In[ ]:


def predictImg(imgpath):
    predict_image = image.load_img(imgpath, target_size = (64,64))
    predict_image = image.img_to_array(predict_image)
    predict_image = np.expand_dims(predict_image, axis=0)
    result = classifier.predict(predict_image)
    return result.max()


# In[ ]:


predictImg('../input/pothole-image-prediction/prediction_data/road1.jpeg')


# In[ ]:


predictImg('../input/pothole-image-prediction/prediction_data/road2.jpg')


# In[ ]:


predictImg('../input/pothole-image-prediction/prediction_data/road3.jpg')


# In[ ]:


predictImg('../input/pothole-image-prediction/prediction_data/road4.jpeg')


# In[ ]:


predictImg('../input/pothole-image-prediction/prediction_data/road5.jpg')

