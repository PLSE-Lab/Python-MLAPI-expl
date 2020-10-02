#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd


# reading files

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sub=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# Separating the data

# In[ ]:


Y=train["label"]
X=train.drop(columns=['label'])


# In[ ]:


print(X.shape)
print(Y.shape)


# Normalizing

# In[ ]:


X = X / 255.0
test = test / 255.0


# Reshaping

# In[ ]:


X = X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]) 
# 
# Reference from : https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# 

# In[ ]:


from keras.utils.np_utils import to_categorical
Y = to_categorical(Y, num_classes = 10)


# Splitting data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


# ### Modelling
# using CNN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


# In[ ]:


#Model 1
model = Sequential()
#Layer 1
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#Layer2
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
#Layer3
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history=model.fit(X,Y,validation_split=0.2,epochs=40)


# ### After train visualizations

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Complexity Graph:  Training vs. Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy Graph:  Training vs. Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()


# ### With Augmentation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=False, 
                             samplewise_center=False, 
                             featurewise_std_normalization=False, 
                             samplewise_std_normalization=False, 
                             zca_whitening=False, 
                             zca_epsilon=1e-06, 
                             rotation_range=10, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             brightness_range=None, 
                             shear_range=0.1, 
                             zoom_range=0.15, 
                             channel_shift_range=0.0, 
                             fill_mode='nearest', 
                             cval=0.0, 
                             horizontal_flip=False, 
                             vertical_flip=False, 
                             rescale=None, 
                             preprocessing_function=None, 
                             data_format=None, validation_split=0.0, dtype=None)


# In[ ]:


datagen.fit(X)


# In[ ]:


#model.compile(optimizer = "Nadam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[ ]:


#history=model.fit(X,Y,validation_split=0.2,epochs=40)


# ### After train visualizations

# In[ ]:


#plt.figure(1)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model Complexity Graph:  Training vs. Validation Loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validate'], loc='upper right')

#plt.figure(2)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model Accuracy Graph:  Training vs. Validation accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validate'], loc='upper right')
#plt.show()


# ### Predicting

# In[ ]:


pred = model.predict(test)
pred = np.argmax(pred,axis = 1)
pred = pd.Series(pred,name="Label")


# In[ ]:


out = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)
out.to_csv("output.csv",index=False)


# ### Help me increase the accuracy :)

# In[ ]:




