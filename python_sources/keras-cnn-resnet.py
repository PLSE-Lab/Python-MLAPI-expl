#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import keras 
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Activation, Flatten, Dense


# In[ ]:


data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
print("Dataset ready")


# In[ ]:


#converts to numpy array
data_train=data_train.as_matrix()
data_test=data_test.as_matrix()


# In[ ]:


#look at the shape
data_train.shape
#data_test.shape 


# In[ ]:


#x is lable y is data in this case, but convention other way around
x=data_train[:,0:1]
#our model expects one-hot-vector
x=to_categorical(x)
x.shape


# In[ ]:


y=data_train[:,1:]
print(y)
#look at shape
y.shape


# In[ ]:


#last dimension is a dummy stand in for red/green/blue channel
y=y.reshape(42000,28,28,1)
y.shape

data_test=data_test.reshape(28000,28,28,1)
data_test.shape


# In[ ]:


model=keras.models.Sequential()

#layer 1
model.add((keras.layers.Conv2D(filters=64, kernel_size=3,input_shape=(28,28,1))))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#layer 2
model.add((keras.layers.Conv2D(filters=32, kernel_size=3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

          
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(.001),metrics=['accuracy'])


# In[ ]:


model.fit(y,x,validation_split=0.2, epochs=5)


# In[ ]:


results=model.predict_classes(data_test)


# In[ ]:


submission = pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})

submission.to_csv("submision.csv", index=False)

print("Prediction Completed")

