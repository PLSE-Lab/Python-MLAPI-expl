#!/usr/bin/env python
# coding: utf-8

# # CNN for MNIST digits classification (99.285% test accuracy, top 25%)
# 
# In this notebook I will use a modified LeNet5 implementation using tensorflow keras API for the problem of handwritten digits classifcation. This is tribute to he man who created the data set and the performance of this CNN is very satisfying for not an acceptable number of parameters.

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
from keras.callbacks import ReduceLROnPlateau
from scipy import ndimage
import matplotlib.pyplot as plt
import time

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Loading the data
# 
# I start by loading the data onto the notebook by using Pandas as the format is CSV.

# In[ ]:


data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# Then I quickly visualize the five first rows of the data using the head() method from pandas

# In[ ]:


data.head()


# ## Convert the panda dataframe into a numpy array
# 
# Convert the panda dataframe to a numpy array. Shuffle the data using a sklearn tool and show a random example image.

# In[ ]:


data_as_np = data.to_numpy()
labels = data_as_np[:,0].reshape((42000,1)) #To avoid having issues with arrays of np shape (42000,)
pixels = data_as_np[:,1:].reshape((42000,28,28,1)) / 255

labels, pixels = shuffle(labels, pixels)

r = int(np.random.uniform(0,41999))
digit=pixels[r,:,:,0] * 255 #pyplot expects a 2D array for a gray scale image
plt.imshow(digit, cmap=plt.cm.gray)   


# ## Split the data 
# 
# We want to split the data we have into a training set and a crossing validation set. I will not use a test set in this application as we do not have a lot of data and we want to maximise the potential of what we have.
# 
# The split ratio is determine by the input test_size.
# 

# In[ ]:


X_train, X_CV, y_train, y_CV = train_test_split(pixels, labels, test_size = 0.08 )

print(f'Training data: \nX_train shape : {X_train.shape} and y_train shape : {y_train.shape}\n')
print(f'Cross validation data: \nX_CV shape : {X_CV.shape} and y_CV shape : {y_CV.shape}')


# ## Time to build the model v2
# 
# For this second version of my submission of the MNIST competition i will build a CNN similar to the LeNet5. Using tensorflow and Keras API. In the second model, I tweaked a bit the architecture to get better performances. I added dropout layers for example.

# In[ ]:


inputs = keras.Input(shape = (28,28,1))

conv1 = layers.Conv2D(16, (8,8), strides = (1,1), activation = 'relu', padding='same')

x = conv1(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (5,5), strides = (1,1), activation = 'relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3,3), strides = (1,1), activation = 'relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Dropout(rate = 0.1)(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), strides = (1,1), activation = 'relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)

x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(120, activation = 'relu')(x)
x = layers.Dropout(rate = 0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(84, activation = 'relu')(x)

outputs = layers.Dense(10, activation = 'softmax')(x)

model = keras.Model(inputs = inputs, outputs = outputs, name="MNISTv2")



model.summary()


model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
             optimizer = keras.optimizers.Adam(lr = 0.001),
             metrics = ['accuracy'])


# ## Learning rate annealer
# 
# I use the Keras method ReduceLROnPlateau to reduce the learning rate when the accuracy on the validation set doesn't increase anymore.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.00001)


# ## Training the model using Keras data augmentation API
# 
# In this second version, I also added a data augmentation to increase the training of my model. I have done so with the Keras preprocessing API. I train the model with real-time data augmentation.

# In[ ]:


datagen = preprocessing.image.ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    zca_whitening=False)

datagen.fit(X_train)

epochs=2
batch_size=10000
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
                    validation_data=(X_CV, y_CV),
                    callbacks=[learning_rate_reduction])


# ## Cross validate the model on an unknown set
# 
# Now that our model is trained with a good accuracy on the training dataset. I improved the performance on the dev set thanks to the added regularisation, droupout layers, as suggested in the previous notebook.

# In[ ]:


test = model.evaluate(X_CV, y_CV)


# ## Remarks
# 
# Now we see that the t dev set accuracy is better than our train set. Thus that mean that we can fit better the training set. There multiple solutions for this:
# * Train a deeper network
# * Train for longer
# * Generate even more data

# ## Load the test set to prepare the submission

# In[ ]:


test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_as_np = test.to_numpy()

pixels_test = test_as_np.reshape((28000,28,28,1)) / 255

tic = time.time()
result = model.predict(pixels_test, batch_size=32)
toc = time.time()

elapsed = (toc-tic)*1000
print(f'The inference time for the 28 000 examples batch size is {elapsed} ms')


result = result.argmax(axis=1).reshape((28000,1))


# ## Prediction on a single example
# 
# We can compute the inference time on a single example and check if our model work properly. Also, it is time to enjoy our work and see that we have done a good job at classifying handwritten numbers.

# In[ ]:


i = int(np.random.uniform(0,27999))

tic = time.time()
inference = model.predict(pixels_test[i:i+1,:,:,:])
toc = time.time()

elapsed = (toc-tic)*1000

print(f'The result is {inference.argmax(axis=1)} and the inference time is {elapsed} ms')

#print(np.average(pixels_test[i:i+1,:,:,:]))
#print('this is the shit \n',np.average(pixels_test[i,:,:,0])) #Theses two lines can be used to make sure we are looking at the same example.

digit=pixels_test[i,:,:,0] * 255 
plt.imshow(digit, cmap=plt.cm.gray)  


# ## Postprocessing
# 
# We need to convert the results in a csv files for submission for the kaggle competition. The exact shape should have the id of the test exmaple and the class predcited.

# In[ ]:


a = np.arange(1,28001).reshape((28000,1))
test_result = np.concatenate((a, result), axis = 1)

print(test_result)


# ## Save as a CSV file
# 
# Finally, we have the data well ordered and everything is ready to create and submit the CSV file.

# In[ ]:


np.savetxt('test.csv', test_result, fmt='%i', delimiter=',', header="ImageId,Label", comments='')


# In[ ]:


check = pd.read_csv('/kaggle/working/test.csv')
check.head(10)


# ## References
# 
# Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, november 1998.
