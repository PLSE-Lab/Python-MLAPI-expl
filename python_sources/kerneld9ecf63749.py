#!/usr/bin/env python
# coding: utf-8

# # Malaria detection
# 
# The purpose of the report is to recognize if a cell is infected or not with malaria disease, with this recognition deaths can be prevented by giving the appropriate treatment on time.
# 
# Malaria is caused by a parasite that is transmitted to humans through the bite of infected anopheles mosquitoes. After infection, parasites travel through the bloodstream to the liver, where the form of parasites appears, called merozoites. The parasites entered the bloodstream and infected the red blood cells.
# 
# The parasites multiply inside the red blood cells, which break after 48 to 72 hours, infecting more red blood cells. The first symptoms usually appear 10 days to 4 weeks after infection. The symptoms appear in cycles of 48 to 72 hours.
# 
# 
# This disease is a major health problem in a large part of tropical and subtropical countries. The Centers for Disease Control and Prevention of the United States estimated that each year there are 300 to 500 million cases of malaria and that more than 1 million people die because of is. 
# 
# In this report is used CNN (convolutional neural networks). This neural networks are typically used for problems like facial recoginition, and image recognition.
# 
# 

# In[25]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[26]:


import cv2
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils


# In[24]:


import tensorflow as tf

tf.__version__


# In[27]:


parasitized_data = os.listdir('../input/cell_images/cell_images/Parasitized/')
print(parasitized_data[:10]) 

uninfected_data = os.listdir('../input/cell_images/cell_images/Uninfected/')
print('\n')
print(uninfected_data[:10])


# # Data
# This dataset was provided with two types, cells that are not infected and cell that are infected.
# 
# The purpose is to recognize wich image has malaria or not.
# 
# In this images below can clearly be seen cell with the parasite.

# In[28]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread('../input/cell_images/cell_images/Parasitized' + "/" + parasitized_data[i])
    plt.imshow(img)
    plt.title('PARASITIZED : 1')
    plt.tight_layout()
plt.show()


# In this four images can be seen how cells are not infected.

# In[29]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread('../input/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i+1])
    plt.imshow(img)
    plt.title('UNINFECTED : 0')
    plt.tight_layout()
plt.show()


# The images need to be re-sized so that all the images have the same size. In addition, it is necessary to take into account to put a size where it can be recognized if there is cell or not with disease that is why 50 is the size chosen.

# In[30]:


data = []
labels = []
for img in parasitized_data:
    try:
        img_read = plt.imread('../input/cell_images/cell_images/Parasitized/' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(1)
    except:
        None
        
for img in uninfected_data:
    try:
        img_read = plt.imread('../input/cell_images/cell_images/Uninfected' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(0)
    except:
        None


# In[31]:


plt.imshow(data[0])
plt.show()


# In[32]:


image_data = np.array(data)
labels = np.array(labels)


# In[33]:


idx = np.arange(image_data.shape[0])
np.random.shuffle(idx)
image_data = image_data[idx]
labels = labels[idx]


# The dataset is split between train and test so that the model can be tested.

# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 101)


# In[35]:


y_train = np_utils.to_categorical(y_train, num_classes = 2)
y_test = np_utils.to_categorical(y_test, num_classes = 2)


# In[36]:


print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')
print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')
print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')
print(f'SHAPE OF TESTING LABELS : {y_test.shape}')


# # Convolutional neural networks
# 
# ![image.png](attachment:image.png)
# 
# A Convolutional Neural Network is a Deep Learning algorithm which can take in an input image, assign importance to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. 
# 
# A CNN is able to successfully capture the spatial and temporal dependencies in an image through the application of relevant filters. The network can be trained to understand the sophistication of the image better.

# In[37]:


import keras
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras import optimizers


# In[38]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# # Adding layers to the model. 
# 
# Here are specified the layers used in the model. There are three layers where the model goes through convolutional, relu and pooling so that it shrinks the image. 
# 
# After these three layers, there are added the fully connected layers, which grab the compressed image and put the results in a list. 
# 
# the way in which the model is going to predict is first creating a list of values with numbers, some values are going to be higher than others, those numbers are going to be the most important. After predicting a new image is going to compare the new list and the list of the model, if the new list has similar numbers in the most important values is because the cell has malaria. That is how CNN works and its so effective in image recognition.

# In[39]:


model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (50,50,3)))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))


# In[40]:


model.summary()


# In[41]:


model.compile(optimizer = 'sgd',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])


# In[42]:


model.summary()


# In[43]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])


# Fitting the model 

# In[44]:


h = model.fit(x_train, y_train, epochs = 20, batch_size = 32)


# In[48]:


plt.figure(figsize = (18,8))
plt.plot(range(20), h.history['acc'], label = 'Training Accuracy')
plt.plot(range(20), h.history['loss'], label = 'Taining Loss')
#ax1.set_xticks(np.arange(0, 31, 5))
plt.xlabel("Epoch's")
plt.ylabel('Accuracy/Loss Value')
plt.title('Training Accuracy and Training Loss')
plt.legend(loc = "best")


# In[46]:


predictions = model.evaluate(x_test, y_test)


# In[47]:


print(f'LOSS : {predictions[0]}')
print(f'ACCURACY : {predictions[1]}')


# # Results
# As can be shown, with this model using the malaria dataset, can be reach a accuracy of 0.95, predicting really good if a cell has malaria, with a loss of 0.13. In the graph can be seen that as the training accuracy grows, the loss decrease. 

# # Conclusion
# 
# Using the model CNN can conclude that it is posible to recognize if a cell has malaria. 
# 
# Also using this model can prevent deaths in many tropical countries with a good implementation.
# 
# It is recommended to use this model whit a 0.95 of accuracy can predict if cells has malaria.
# 
# For future projects would like to use this model in parts of the countrie with local images to see if can be implemented in the country.
