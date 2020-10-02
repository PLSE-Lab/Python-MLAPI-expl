#!/usr/bin/env python
# coding: utf-8

# # Digit recognization for everyone - 0.99

# ## 1. Importing necessary libaries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# ## 2. Load the train and test data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#exploring the data
train.head()


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# ## 3. Splitting the dependent and independent variables

# In[ ]:


train_images=train.iloc[:,1:].values 
#.values is used to convert dataframe to numpy for better accesibility 


# In[ ]:


train_labels=train.iloc[:,:1].values


# In[ ]:


test_images=test.values


# ## 4. Scaling

# In[ ]:


np.max(train_images)


# In[ ]:


train_images=train_images/255.
test_images=test_images/255.


# In[ ]:


test_images.shape


# In[ ]:


train_images.shape


# ## 5. Visualize the Data

# Before visualize the image data we have to change the dimensional data. because displaying the image data we need to have 3 dimension

# In[ ]:


train_images=train_images.reshape(train_images.shape[0],28,28)
train_images.shape


# In[ ]:


test_images=test_images.reshape(test_images.shape[0],28,28)
test_images.shape


# 1. ### 5.1. Visualize the train image

# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(train_images[1])


# ### 5.2. Visualize the first 25 train images

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
    plt.xticks([])
    plt.yticks([])


# Reshaping the train and test images

# In[ ]:


train_images=train_images.reshape(train_images.shape[0],28,28,1)
test_images=test_images.reshape(test_images.shape[0],28,28,1)


# ## 6. CNN Deep learning model building

# ### 6.1. Initialize the CNN model

# In[ ]:


classifier=Sequential()


# ### 6.2. Create Convolution layer

# Feature detector: Filter, Kernel, or Feature Detector is a small matrix used for features detection.
# 
# Convolutional layer compares the raw data against the feature detector and store the count of matched 1's in the feature map. This reduces the no of inputs.
# 
# ![image.png](attachment:image.png)
# 
# After the the convolution, We apply rectified linear unit (ReLU) inorder to remove linearity or increase non linearity in our images

# In[ ]:


classifier.add(Conv2D(32,(3,3), input_shape=(28, 28,1), activation='relu'))


# ### 6.3. Create MaxPooling Layer

# Pooling layer is used to capture distinctive feature of an object even if the image is rotated, tiled, posing different pose, different lightining

# In[ ]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:


classifier.add(Flatten())


# ### 6.4. Creating hidden layer

# In[ ]:


classifier.add(Dense(activation='relu',units=128))


# ### 6.5. Creating output layer

# In[ ]:


classifier.add(Dense(units=10,activation='softmax'))


# ### 6.6. Compile the CNN

# In[ ]:


classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# ### 7. Data Augmentation

# Data Augmentation is technique of creating slightly different or new images of your input image. Data Augumentation avoids overfitting. Model performs well when we create more training images.
# 
# There are many ways to perform data augmentation,
# * Cropping
# * Rotating
# * Scaling
# * Translating
# * Flipping

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)


# In[ ]:


from sklearn.model_selection import train_test_split
X = train_images
y = train_labels
train_images1, X_val, train_labels1, y_val = train_test_split(train_images, train_labels, test_size=0.10, random_state=42)
batches = datagen.flow(train_images, train_labels, batch_size=64)
val_batches=datagen.flow(X_val, y_val, batch_size=64)


# In[ ]:


history=classifier.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=2, 
                    validation_data=val_batches, validation_steps=val_batches.n)


# ### 8. Submitting the predictions

# In[ ]:


predictions = classifier.predict_classes(test_images)


# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv('digit_submission3.csv', index=False, header=True)


# You can also increase the number of epochs for better predictions. But it also increases the execution time
