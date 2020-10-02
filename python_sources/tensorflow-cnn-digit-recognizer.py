#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Convolutional Neural Network: Digit Recognizer
# > According to [Wikipedia](https://en.wikipedia.org/wiki/Handwriting), the handwriting is characterized by factors such as: shape of letters, spacing between letters, slope of letters, arrhythmia, pressure applied and size/thickness of the letters.

# TABLE OF CONTENTS
# 0. OVERVIEW
# 1. IMPORT LIBRARIES
# 2. DATA PRE-PROCESSING
# 3. CNN MODEL
# 4. PREDICTING RESULTS
# 5. CONCLUSIONS

# # 0. OVERVIEW
# The training dataset is a csv-file with 785 columns: 1 label column and 784 pixel columns. The csv file is made of images, which are 28 pixels x 28 pixels, so each image includes in total 784 different pixels. Each column represents the individual pixel value. The pixel value contain data on the exact intensity of the gray scale color between 0 to 255.  Each row on the csv-file represents a single image. The testing dataset is similar to the training, but without the label of correct digit. 
# 
# The objective is to correctly classify labels of the testing dataset. To achieve our objective, we will create Convolutional Neural Network with the Tensorflow - to classify images of digits into specific class. Prior to creating the Tensorflow CNN, we will import libraries required and the dataset. The star of this notebook is the Convolutional Neural Network, which you can read details in [my Medium profile.](https://medium.com/@tmmtt) The conclusions section will discuss in detail, the potential further improvements, based the metrics applied.
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
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator


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
train_y = to_categorical(train_y)
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
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu', input_shape = (28,28,1)),
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu'),
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2), 
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation = "relu"),
tf.keras.layers.Dense(10, activation = "softmax")
])


# # 4. PREDICTING RESULTS
# Now, we will print the model summary, compile the model with Adam-optimizer.
# Finally, let's apply augmented data and fit the data.

# In[ ]:


model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# Data augmentation
# A useful site, where we can learn more about Data augmentation using Keras [ImageDataGenerator().](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

# In[ ]:


datagen = ImageDataGenerator(rotation_range=5, zoom_range=0.09) # try  
datagen.fit(train_X)
batch =512
model.fit_generator(datagen.flow(train_X, train_y, batch_size=batch), epochs=45)


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
pred_digits.head()


# # 5. CONCLUSIONS
# 
# There is massive amount of Kagglers with super accurately performing classifiers. If we look at the leaderboard: there are hundreds of users, who are capable of reaching close to human level performance. So, there arises the question, if we are able to classify 100% accurately in the future?
# 
# There appears certainly lot of research papers, which Kagglers are familiar. Further, the transfer learning is a great example, how we can improve our model performance. Within time, we will as well be able to put even more computer resources, to calculate even more complex networks.
# 
# However, it seems still early to say, that we are certainly going to reach the 100% accuracy. To efficiently overcome this challenge, we will need to share more about best practices, which help objectively to tune any model, to a better one. 
# 
# For example, it is interesting to read - lot of notebooks offer really high accuracy, but relatively less offer ideas: How their models could be further improved? Here, it is obvious - the importance of a good metric. We see lot of notebooks, where the analysis of wrong results is left out completely. There needs to be a clear metric, which allows us to know: are we moving to right direction. But beside knowing what was right or wrong: the next step is to analyze the concrete errors made by the model. 
# 
# The data augmentation is a great point of analysis. The possibility of adding data is great: few lines of extra code for data augmentation gives significant boost to our model accuracy. However, it leaves us to wonder: Is there still major room for further improvements by adding further data augmentation methods? Is there a greater chance to improve our model by making our CNN more complex or making our dataset more complex? 
# 
# 
# 
# 
# 
# 
# 

# 
