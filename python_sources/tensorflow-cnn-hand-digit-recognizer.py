#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Convolutional Neural Network: Digit Recognizer
# > According to [Wikipedia](https://en.wikipedia.org/wiki/Handwriting), the handwriting is characterized by factors such as: shape of letters, spacing between letters, slope of letters, arrhythmia, pressure applied and size/thickness of the letters. 
# 

# **TABLE OF CONTENTS**
# 
# 0. INTRODUCTION
# 1. IMPORT LIBRARIES
# 2. DATA PRE-PROCESSING
# 3. CNN MODEL
# 4. PREDICTING RESULTS
# 5. CONCLUSIONS

# # 0. INTRODUCTION
# 
# 
# **The training dataset** is a csv-file with 785 columns: 1 label column and 784 pixel columns. The csv file is made of images, which are 28 pixels x 28 pixels, so each image includes in total 784 different pixels. Each column represents the individual pixel value. The pixel value contain data on the exact intensity of the gray scale color between 0 to 255.  Each row on the csv-file represents a single image. The testing dataset is similar to the training, but without the label of correct digit. 
# 
# **The objective** is to correctly classify labels of the testing dataset. In the past, there has been several attempts to classify hand written digits using different architectures e.g. SVM, neural networks, K-Nearest Neighbour and of course CNNs.  Some of these prior works are listed along with the MNIST-database website, [listed here](http://yann.lecun.com/exdb/mnist/). Here, we will create Convolutional Neural Network with the Tensorflow - to classify images of digits into specific class. Prior to creating the Tensorflow CNN, we will import libraries required and the dataset. The star of this notebook is the Convolutional Neural Network, which you can read details in [my Medium profile.](https://medium.com/@tmmtt) The conclusions section will discuss in detail, the potential further improvements, based the metrics applied.
# 
# **Data augmentation**
# In this notebook, we turned into data augmentation. There is a useful information, which offers details of further ways to apply data augmentation with [Keras ImageDataGenerator class.](https://keras.io/preprocessing/image/)

# # 1. IMPORT LIBRARIES
# Let's start by importing the required Python libraries. We will use general purpose data handling libraries: numpy, pandas and matplotplip for plotting. The Tensorflow and Keras libraries are used for building the CNN-moodel and data augmentation.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# # 2. DATA PRE-PROCESSING
# To apply the Convolutional Neural Network, we wil import the csv-data using pandas read_csv. We will split the correct labels

# In[ ]:


#Training data
train_data = pd.read_csv('../input/train.csv') # Import the dataset
train_y = train_data["label"] # Create label vector
train_data.drop(["label"], axis=1, inplace=True) # Remove the label vector from the pixel column matrix
train_X = train_data
train_X = train_X.values.reshape(-1, 28, 28, 1)
train_y = train_y.values
train_y = tf.keras.utils.to_categorical(train_y)
 
train_X = train_X/255.00 # Normalization
#Test data
test_X = pd.read_csv('../input/test.csv')
test_X = test_X.values.reshape(-1,28,28,1)
test_X = test_X / 255.0 # Normalization


# # 3. CNN MODEL
# Our tensorflow model consist:
# * Two CNN layers, Maxpooling, dropout, Flatten, Dense, Output layer

# In[ ]:


model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation ='relu', input_shape = (28,28,1)),
tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation ='relu'),
tf.keras.layers.Dropout(0.7),
tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation ='relu'),
tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation ='relu'),
#tf.keras.layers.Dropout(0.7),    
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
tf.keras.layers.Conv2D(32, kernel_size = (7,7), padding = 'same', activation ='relu'),
tf.keras.layers.Conv2D(32, kernel_size = (7,7), padding = 'same', activation ='relu'),
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(1024, activation = "relu"),
tf.keras.layers.Dense(256, activation = "relu"),
tf.keras.layers.Dense(10, activation = "softmax")
])


# # 4. PREDICTING RESULTS
# Now, we will print the model summary, compile the model with Adam-optimizer.
# Finally, let's apply augmented data and fit the data.

# In[ ]:


model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# **Data augmentation**
# A useful site, where we can learn more about Data augmentation using Keras [ImageDataGenerator().](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). The core concepts are the Data Augmentation (DA) Iterator, which is first created, where we define the way the data will be augmented. Then, we fit the Iterator with the internal statistics from the training data. Final step is to fit the CNN-model, by starting to feed batches of images using our Data Augmentation (DA) Iterator.
# 

# In[ ]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=43, zoom_range=0.24)# Create Data Augmentation (DA) Iterator
datagen.fit(train_X) #Our DA Iterator is trained with the image data, to calculate internal statistics
# Let's add callbacks, which adjust learning rate
ln_fc = lambda x: 1e-3 * 0.99 ** x
lrng_rt = tf.keras.callbacks.LearningRateScheduler(ln_fc)
# Fit our CNN-model using the DA Iterator using Flow-method, which feeds batches of augmented data:
digitizer = model.fit_generator(datagen.flow(train_X, train_y, batch_size=1024), epochs=80, callbacks=[lrng_rt]) 


# Our results

# In[ ]:


predictions = model.predict(test_X)
predictions[354]
pred = np.argmax(predictions, axis=1)

plt.imshow(test_X[354][:,:,0],cmap='gray')
plt.show()

pred[354]

pred_digits = pd.DataFrame({'ImageId': range(1,len(test_X)+1) ,'Label':pred })
pred_digits.to_csv("pre_digits.csv",index=False)


# **Accuracy plotted**
# Let's plot the mode accuracy during each epoch:

# In[ ]:


plt.plot(digitizer.history['acc'])
plt.show()


# # 5. CONCLUSIONS
# 
# There is massive amount of Kagglers with super accurately performing classifiers: its really impressive to see such great people collaborating within this platfor. If we look at the leaderboard: there are hundreds of users - capable of reaching "close to/above" human level performance. Thus, the obvious question arises: Are we able to classify 100% accurately in the future hand written digits?
# 
# Based on such high quality work within Kaggle: it is not surprising, that the users have been reading the research papers within this field of study.  New research papers within image recognition - are potential sources for able to reeach 100% accuracy. For exmaple transfer learning - is a great example: how we can improve our model performance. The availability of even better computer resources, may open a path for being able to train for even more complex neural networks for this task. However, it seems too early to state, that we will reach 100% accuracy.
# 
# To efficiently gain higher performance - we will need to further promote "best practices", which help objectively to tune any model, to a better one. It is interesting to read notebooks within this platform and rather often written with very interesting and complext ideas for the model itself, but relatively less importance on analysing the metrics. There appears abudance of models compared to lack of ideas for further improve them. Its true, that Kagglers offer great solution, but we tend to analyze less: where our model has gone wrong. 
# 
# **Example: Confusion matrix:**
# 
# Its rather typical to see on many occasions the confusion matrix included to the workbook. It is great metric: (1) An unique metric to further improve our model (2) it deals separately performance of each digit category. If we stay laser-focused on this metric, our objective is to discover the causes, that prevent us getting correctly classified digits in situations, where even a human could make an error. Data augmentation may be helpful on this approach, but our focus will be particularly to address situations e.g. digit 1 is mixed with 4 or 7, rather than any of the other digits. Is it likely, that a human is able to differentiate 1 and 7 in these missclassified situations? 
# 
# There are many Kaggle workbooks with very high accuracy and no other metric and no analysis of results. In fact, lot of workbooks offer no ideas at all for potential further improvement: What our model is not doing well vs. what our model is doing extremely well? For example, we had a good sense of ideas to make a simple, but efficient network, which trained well on high number of epochs. It was only once we had worked on tuning the model parameters and data augmentation, that it made sense to learn on using the Tensorflow LearningRateReducer.
# 
# Data augmentation is a great point of further analysis: few lines of extra code, offered a significant boost to the model performance. It was the single most important improvement, which was made within this notebook. It still leaves us wondering, if further improvements are possible by using different type of data augmentation methods to address the dimension or there is a better approach to address error using confusion matrix? 
# 

# 
