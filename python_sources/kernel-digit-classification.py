#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/yash0304/0304first/blob/master/digit_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #Digit Recognition Using Tensor Flow and Keras

# ### Importing Libraries

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ### Importing MNIST Data Set for Digits Data

# In[ ]:


mnist = tf.keras.datasets.mnist


# ### Training and Testing Data Set

# In[ ]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# In[ ]:


x_train[0]


# ### Normalize

# In[ ]:


x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)


# In[ ]:


x_train[0]


# In[ ]:


x_test[0]


# ### Model Building and Training

# In[ ]:


model = tf.keras.Sequential()


# In[ ]:


model.add(tf.keras.layers.Flatten()) # adding flatten curve


# In[ ]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #relu used for adding Non Linearity


# In[ ]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #adding dense layer


# In[ ]:


model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # output as softmax activation method


# ### Compiling the model
# 

# In[ ]:


model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ['accuracy']
              )
model.fit(x_train,y_train,epochs=10)


# ### Validating the model

# In[ ]:


validation_loss,validation_accuracy = model.evaluate(x_test,y_test)


# In[ ]:


print(validation_loss,validation_accuracy)


# ### Saving the model

# In[ ]:


model.save("firstModel.model")


# ### Loading the model

# In[ ]:


new_model = tf.keras.models.load_model("firstModel.model")


# ### Predicting with testing data set

# In[ ]:


prediction = new_model.predict(x_test)


# In[ ]:


prediction[0]


# ###Function to show the Output

# In[ ]:


def show_numbers(i):
  plt.imshow(x_test[i])
  print("The Number shown in the image is:   ",np.argmax(prediction[i]), end='\n')


# In[ ]:


show_numbers(30)


# In[ ]:




