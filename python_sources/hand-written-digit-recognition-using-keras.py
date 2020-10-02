#!/usr/bin/env python
# coding: utf-8

# # Hand-Written Digit Recognition using MNIST dataset and Keras 
# 
#     
# 
# > #### *Often referred as the "Hello World" of object recognition for Machine Learning and Deep Learning. *
# > 

# ![Image1](https://www.researchgate.net/profile/Alessandro_Di_Nuovo/publication/328030580/figure/fig1/AS:677340703121411@1538502016731/Examples-handwritten-digits-in-the-MNIST-dataset.ppm)

# ### In this tutorial, I have broken down the process into 6 easy steps:
# 1. Loading libraries and MNIST Dataset
# 2. Dividing training and Testing Data
# 3. Normalizing and Flattening the data.
# 4. Adding layers and compiling the model
# 5. Evaluating the model
# 6. Validation of the model.

# #### This problem is great if you are a beginner and want to explore the world of deep learning..

# ## 1] Firstly, we'll import the libraries and dataset:
# 
# 

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt #to visualize an image from our dataset


# **What is Tensorflow?**
# * Is an open source artificial intelligence library, using data flow graphs to build models.
# * It allows developers to create large-scale neural networks with many layers. 
# 
# TensorFlow is mainly used for: Classification, Perception, Understanding, Discovering, Prediction and Creation.

# Now as the libraries are imported, let's import our dataset as well:

# In[ ]:


mnist=tf.keras.datasets.mnist  #28x28 image of hand-written digits


# ## 2] Dividing training and testing data

# In[ ]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# **But, what does MNIST dataset contain?**
# * Almost 70,000 images of hand-written digits from 0-9 
# * Every image is composed of 28x28 pixels, i.e. 784 pixels per image
# * Each pixel contains the intensity of color on grayscale i.e from black to white i.e. 0-255
# 
# 
# [Click here if you want to understand the MNIST dataset and Neural Networks with more ease.](https://www.youtube.com/watch?v=aircAruvnKk)
# 
# 

# **Before jumping on to the next step, let's see how an image from our dataset looks:**

# In[ ]:


plt.imshow(x_train[0],cmap=plt.cm.binary)


# *As you can see here, the image is composed of 28x28 pixels, and each pixel contains a value from 0-255 stored in it.*
# 
# 
# 
# **Why 0-255?**
# 
# Here's the reason:
# 
# (RGB representation depicts color in 8-bit format, 2^8=256 i.e. 0-255 unique values, hence the unique colors)
# 
# **But, here we are using only grayscale color, so to normalize the data in a range of 0 to 1,
# Let's jump on to the next step:
# **

# ## 3] Normalizing and Flattening of data

# In[ ]:


#It's easier for our network to learn this way

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

#Now, the data has been normalized


# In[ ]:


#The image will look a bit dull as each pixel stores a value between 0-1, but don't worry..

plt.imshow(x_train[0],cmap=plt.cm.binary)


# In[ ]:


#Loading the Sequential model

model=Sequential() 


# In[ ]:


# data is in multidimensional form,so we need to make it into simpler form by flattening it:

model.add(Flatten())


# 
# 
# 
# ## 4] Adding layers and compiling the model

# In[ ]:


model.add(Dense(128,activation="relu"))  #128 neurons, activation function to fire up the neuron
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax")) 

#10 because number of classification is from 0-9 i.e. 10 different types of data


#softmax for probability distribution


# In[ ]:



#parameters for training of the model:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])     

#loss is the degree of error


# In[ ]:



#neural networks don't actually try to increase accuracy but tries to minimize loss
#epochs are just number of iterations the whole model will follow through

model.fit(x_train,y_train,epochs=3)


# ## 5] Evaluating the Model

# In[ ]:


val_loss,val_acc=model.evaluate(x_test,y_test)
print("\n\n\n\tEvaluation of the Model:\n")
print("Value Accuracy=",val_acc,"\nValue Loss=",val_loss)


# Moving on to the last and final step:

# ## 6] Validation of the Model

# In[ ]:


#Saving the model as Number_Reader
    
model.save('Number_Reader.model') 


# In[ ]:


#importing our Number_Reader.model into new_model

new_model = tf.keras.models.load_model('Number_Reader.model')


# In[ ]:


pred=new_model.predict(x_test)
print(pred)  #these are all probability distributions


# It looks a bit scary ..but, the outcome is wayyy simpler.
# Let's have a look at what all these values are trying to convey.
# But, we'll be needing numpy library for this.

# In[ ]:


import numpy as np


# In[ ]:


print("MODEL OUTPUT: ")
print(np.argmax(pred[0])) 

#takes out the maximum value, from the probability density given above

Here, our model gave number 7 as the output.
To verify the image was really of number 7 or not, let's find out by plotting the image:
# In[ ]:


print("The Actual Image was of:")
plt.imshow(x_test[0])
plt.show()


# **Finally, our model's classification matches the outcome! **
# 
# **Therefore, the model is valid.**
