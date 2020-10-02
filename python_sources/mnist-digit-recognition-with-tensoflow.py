#!/usr/bin/env python
# coding: utf-8

# # MNIST Digit Recognization

# ![](http://i.imgur.com/E07YqHb.png)

# ### Index
# 1. Preprocessing data
# 2. Creating Model
# 3. Creating Sample Submission File
# 4. Conclusion

# MNIST (Modified National Institute of Standards and Technology) is the hello world program for the Computer Vision. This is my second competition and first kernel using Tensorflow. In this Kernel I have used tensorflow sequential model to predict the number from the image.

# ### 1. Preprocessing data

# In[ ]:


#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization library
from sklearn.model_selection import train_test_split # training and splitting data

import tensorflow as tf #tensorflow

import os
print(os.listdir("../input"))


# We have total three csv files in this dataset. As the name suggests, `train.csv` and `test.csv` contains training images and testing images respectively. We have `sample_submission.csv` file to create csv file for submission. So here we loading the data for training and testing images.

# In[ ]:


#loading training and testing dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.columns


# In[ ]:


test_df.columns


# We have `28*28` = `784` pixels of the images. In `training.csv` we have the correct digit of the number. We will create model based on the `training.csv` file and will submit the output of the predicted values on `testing.csv` file.

# In[ ]:


# seperating labels and images [X_train = images, Y_train = numbers on respective image]
X_train = train_df.drop(labels = ["label"],axis = 1) # contains values of digits in 255 range
Y_train = train_df['label'] # contains digits
X_train = X_train.values.reshape(-1,28,28,1)/ 255 # reshaping arrays in tensors


# In[ ]:


# creating common method to display image
def displayImage(image):
    plt.imshow(image[:,:,0], cmap=plt.cm.binary)
    
def displayImageWithPredictedValue(image, prediction):
    print('Predicted output image is ', np.argmax(prediction))
    plt.imshow(image[:,:,0], cmap=plt.cm.binary)


# In[ ]:


# displaying first first value
displayImage(X_train[0])


# ### 2. Creating Model

# I am using [Sequential model](https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential) and used **Flatten** layer to convert tensors into arrays. Using [relu activation](https://www.tensorflow.org/api_docs/python/tf/nn/relu)(REctified Linear Units) with different input image parameters, I degraded features vectors to 64. Lastly, I used [Softmax function](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) with 10 output entries (0 to 9). I compiled model with adam optimzer and used loss function as sparse_categorical_crossentropy. At the end I trained model using data with 2 epochs. Epoch is training loop (forward and backward) to train model.

# In[ ]:


model = tf.keras.models.Sequential() # creating Sequential model
model.add(tf.keras.layers.Flatten()) # flattening the input arrays
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # using relu activation function
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu)) # using relu activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # activation function to get number of output

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compiling model

model.fit(X_train, Y_train.values, epochs=5) # training model and fitting data


# In[ ]:


model.summary()


# In[ ]:


# splitting data to evalueate model
X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                              Y_train, 
                                              test_size=0.20,
                                              random_state=42,
                                              shuffle=True,
                                              stratify=Y_train)


# In[ ]:


val_loss , val_accuracy = model.evaluate(X_val, Y_val) # evaluating performance of the model
print(val_loss, val_accuracy)


# In[ ]:


predictions = model.predict([X_val])
displayImageWithPredictedValue(X_val[12], predictions[12])


# In[ ]:


test_df = test_df.values.reshape(-1,28,28,1)/255


# In[ ]:


predictions = model.predict([test_df])
displayImageWithPredictedValue(test_df[10], predictions[10])


# ### 3. Creating Sample Submission File

# In[ ]:


# creating array of outputs, to add into submission.csv file
results = np.argmax(predictions,axis = 1)
# creating submission.csv file
submission = pd.DataFrame(data={'ImageId': (np.arange(len(predictions)) + 1), 'Label': results})
submission.to_csv('submission.csv', index=False)


# ### 4. Conclusion:
# 
# I created model that can predict the correct number values from the images with accuray 0.96614. This dataset can be further explored by applying classification methods such as SVM and K-nearest neighbors.
# 
# You can upvote this kernel to help others to learn it. Thank you :)
