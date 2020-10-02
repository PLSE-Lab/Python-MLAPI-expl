#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading The Dataset

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')
test.head() 


# In[ ]:


y_train = train['label']
X_train = train.drop(['label'], axis=1)


# In[ ]:


plt.figure(figsize=(15,7))
g = sns.countplot(y_train, palette='icefire')
plt.title('Number of digit classes')
y_train.value_counts() 


# In[ ]:


img = X_train.iloc[1]
img = np.asanyarray(img)
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()


# In[ ]:


img = X_train.iloc[2]
img = np.asanyarray(img)
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()


# In[ ]:


img = X_train.iloc[10]
img = np.asanyarray(img)
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()


# # Normalization, Reshape and Label Encoding
# 
# * If we do normalization, CNN works faster. 
# * We need to reshape our data because of Keras. It will be like that 28x28x1 
# * Label Encoding
#      Encode labels to one hot vectors
#         2 => [0,0,1,0,0,0,0,0,0,0]
#         4 => [0,0,0,0,1,0,0,0,0,0]

# In[ ]:


X_train, test = X_train / 255, test / 255 
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)
X_train.head() 


# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
X_train.shape


# In[ ]:


test=test.values.reshape(-1,28,28,1)
test.shape


# In[ ]:


y_train = to_categorical(y_train, num_classes = 10)


# # Train and Test Split
# 
# * We are going to split the data into train and test sets.

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.1, random_state=2)


# In[ ]:


print("x_train shape",x_train.shape)
print("x_test shape",x_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)


# In[ ]:


# Some examples
plt.imshow(X_train[5][:,:,0],cmap='gray')
plt.show()


# # Convolutional Neural Network
# 
# * CNN is used for image detection, image classification. 
# * Convolution Operation: 
#     * We hace feature detector for our images. 
#     * Feature detectors are also called as kernel or filter.
#     * Feature detector detects features like edges or convex shapes. 
#     * If we apply it for an image, we'll have an feature map. This is the result of the our detector. 
#        

# ## Same Padding
# 
# * As we keep applying conv layers, the size of the imagesses' volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features.
# 
# <img src="https://missinglink.ai/wp-content/uploads/2019/03/Frame-2.png" />

# ## Max Pooling
# * It reduces the number of parameters
# * It makes the detection of features invariant to scale or orientation changes.
# 
# <img src="https://preview.ibb.co/gsNYFU/maxpool.jpg" />
# 
# * At the and, this periot repeats a few times or how you would like, and these results will be flattened and will feed into an ANN.

# # Create Model

# In[ ]:


model = Sequential() 

model.add(Conv2D(filters = 8, kernel_size=(5,5), padding='Same', 
                activation='relu', input_shape=(28,28,1))) 

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25)) 

model.add(Flatten())
## ANN

model.add(Dense(256, activation="relu")) 
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# ### Define Optimizer

# In[ ]:


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# ## Compile Model

# In[ ]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


epochs = 10  # for better result increase the epochs
batch_size = 250


# ## Data Augmentation 
# 
# * To avoid overfitting problem, we can expand our handwritten digit dataset
# * Rotate images, zoom images ect 

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)


# # Fit the Model 

# In[ ]:


history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)


# # Evaluate the model

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


Y_pred = model.predict(x_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_val,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Mat")
plt.show()


# In[ ]:




