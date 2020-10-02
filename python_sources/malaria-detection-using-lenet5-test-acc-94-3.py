#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import keras
from PIL import Image


# ## Importing Data, and Data Augmentation
# 
# We first import data from two folders containing images of infected and healthy cells, then we reduce the pixcels to 32 x 32 x 3, then we do some data augumentation (image rotation and image blurring), then we create a list of labels for both healthy and infected cells and concatinates them, then we concatinate infected and healthy cell data into a single large data.

# In[2]:


# Infected Cells
infected = [cv2.imread(file) for file in glob.glob('../input/cell_images/cell_images/Parasitized/*.png')]

# Healthy Cells
healthy = [cv2.imread(file) for file in glob.glob('../input/cell_images/cell_images/Uninfected/*.png')]


# In[3]:


infected[0].shape 


# In[4]:


len(infected)


# In[5]:


len(healthy)


# In[6]:


healthy[0].shape 


# In[7]:


dim = (32, 32)
infected = [cv2.resize(file, dim, interpolation = cv2.INTER_AREA) for file in infected] #reducing pixels to 32 x 32 x 3
healthy = [cv2.resize(file, dim, interpolation = cv2.INTER_AREA) for file in healthy] #reducing pixels to 32 x 32 x 3


# In[8]:


def rotateImage(image, angle):      #function to rotate images
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[9]:


rotated45 = [rotateImage(file, 45) for file in infected]                 #rotating images 45 degrees
rotated75 = [rotateImage(file, 75) for file in infected]                 #rotating images 75 degrees
blurred = [cv2.GaussianBlur(file, (15, 15), 0) for file in infected]     #blurring images

rotated45_healthy = [rotateImage(file, 45) for file in healthy]                 
rotated75_healthy = [rotateImage(file, 75) for file in healthy]         
blurred_healthy = [cv2.GaussianBlur(file, (15, 15), 0) for file in healthy]


# In[10]:


infected = np.concatenate((np.array(infected),np.array(rotated45)))
infected = np.concatenate((np.array(infected),np.array(rotated75)))
infected = np.concatenate((np.array(infected),np.array(blurred)))

healthy = np.concatenate((np.array(healthy),np.array(rotated45_healthy)))
healthy = np.concatenate((np.array(healthy),np.array(rotated75_healthy)))
healthy = np.concatenate((np.array(healthy),np.array(blurred_healthy)))


# In[11]:


infected[0].shape 


# In[12]:


healthy[0].shape 


# In[13]:


len(infected)


# In[14]:


len(healthy)


# In[15]:


y_infected = np.ones(len(infected)) # labels for infected
y_healthy = np.zeros(len(healthy))  # labels for uninfected
y = np.concatenate((y_infected,y_healthy)) # Labels


# In[16]:


len(y)


# In[17]:


X = np.concatenate((np.array(infected),np.array(healthy)))


# In[18]:


X[0].shape


# In[19]:


len(X)


# In[20]:


plt.imshow(X[13103])
print(y[13103])


# In[21]:


plt.imshow(X[98000])
print(y[98000])


# ### Splitting Data into Training, Validation ad Test Sets
# 
#  We split this data along with labels into a training, test and validation set.
#  
#  

# In[22]:


X_train, X_1, Y_train, Y_1 = train_test_split(X, y, test_size = 0.1, shuffle = True)
X_val, X_test, Y_val, Y_test = train_test_split(X_1, Y_1, test_size = 0.5, shuffle = True)


# In[23]:


X_train.shape


# In[24]:


X_val.shape


# In[25]:


Y_train.shape


# In[26]:


Y_val.shape


# In[27]:


X_test.shape


# ### Model

# In[28]:


Batch_size = 256
Num_epoch = 30


# In[29]:


def model():
    model = Sequential()
    
    model.add(Conv2D(6, kernel_size = (3,3),  strides=(1, 1), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(16, kernel_size = (3,3), strides=(1, 1), activation = 'relu'))
    model.add(MaxPool2D (pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(120, activation = 'relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(84, activation = 'relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    
    return model


# In[30]:


model = model()
model.compile(loss=keras.losses.binary_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
model.summary()


# In[31]:


model_train = model.fit(X_train, Y_train, validation_data= (X_val, Y_val), batch_size= Batch_size, epochs= Num_epoch, verbose=1)


# ### Testing on Test Data

# In[32]:



X_loss, accuracy = model.evaluate(X_test,Y_test)
print('\n', 'Test_Accuracy:-', accuracy)


# In[33]:


Y_pred = model.predict(X_test)


# In[34]:


for i in range(len(Y_pred)):
    if (Y_pred[i]<0.5):
        Y_pred[i] = 0
    else:
        Y_pred[i] = 1
Y_pred[500][0]


# In[35]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test , Y_pred))


# ### Training and Validation Curves

# In[36]:


def show_accuracy_vs_epoch(model):
    xlabel= 'Epoch'
    legends = ['Training', 'Validation']
    plt.xlabel(xlabel, fontsize = 10)
    plt.ylabel('Accuracy', fontsize = 10)
    title = 'Model with Data Augmentation'
    plt.title(title)
    plt.plot(model_train.history['acc'])
    plt.plot(model_train.history['val_acc'])
    plt.legend(legends, loc = 'lower right')
show_accuracy_vs_epoch(model_train)


# In[37]:


errors = []

for i in range(len(Y_pred)):
    if (Y_pred[i][0] != Y_test[i]):
        errors.append([i, Y_pred[i][0]])


# In[38]:


def errors_images(errors_ind):
    '''Show images with their predicted and real labels'''
    n = 0
    rows = 3
    cols = 3
    fig, ax = plt.subplots(rows, cols, figsize = (10, 10))
    
    for row in range(rows):
        for col in range(cols):
            index = errors_ind[n][0]
            pred = errors_ind[n][1]
            img = X_val[index]
            ax[row, col].imshow(img)
            ax[row, col].set_title('True:{} pred:{}'.format(Y_test[index], pred))
            n = n+1


# In[39]:


errors_images(errors[0:])


# In[ ]:




