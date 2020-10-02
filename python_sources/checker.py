#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# In[ ]:


data=pd.read_csv('../input/bmi-csvv/data_cut.csv')
names = data['name']
y= data['bmi']
g= pd.get_dummies(data.gender)
y=np.array(y)
y=y[:, None]


# In[ ]:


#This block of code is run and saved in picture_mtcnn.npz. Hence, skip this
l=[]
g=[]
from tensorflow.keras.preprocessing.image import load_img
c=0
for i in range(0, len(names)):
    if(c%50 == 0):
        print(c)  
    img = load_img('../input/imagesright/Images_cut/' + names[i], target_size=(160,160))
    image_array  = tf.keras.preprocessing.image.img_to_array(img)
    #image_array1 = image_array.reshape(1, 160, 160, 3)
    l.append(image_array)
    g.append(y[i])
    flipped = tf.image.flip_left_right(image_array)
    #image_array1 = tf.reshape(flipped, (1, 160, 160, 3), name=None)
    l.append(flipped)
    g.append(y[i])
    flipped = tf.image.adjust_saturation(image_array, 3)
    #image_array1 =tf.reshape(flipped, (1, 160, 160, 3), name=None)
    l.append(flipped)
    g.append(y[i])
    c=c+1


# In[ ]:


from numpy import savez_compressed
savez_compressed('picture_mtcnn.npz', A_new,y)


# In[ ]:


#Run this to load the variables
from numpy import load
data = load('../input/tensorforem/tensor.npz')
X, y = data['arr_0'], data['arr_1']
#X = tf.convert_to_tensor(X)
#X=tf.cast(X, tf.float32, name=None)
X = X.astype(np.float32)
X = (X/127.5) - 1


# In[ ]:


#Binning the different weight categories
y1=[]
for i in y:
    if(i < 18.5):
        y1.append(0)
    elif(i< 25):
        y1.append(1)
    elif (i< 30):
        y1.append(2)
    else:
        y1.append(3)
y = np.array(y1)


# In[ ]:


del y1


# In[ ]:


#Smote oversampling method
X= tf.reshape(X, [4196, 76800])
from imblearn.over_sampling import SMOTE
oversample = SMOTE('not majority')
X, y = oversample.fit_resample(X, y)
X= np.reshape(X, (9452,160,160,3))


# In[ ]:


#One hot encoding
b = np.zeros((y.size, y.max()+1))
b[np.arange(y.size),y] = 1
b = b.astype(int)


# In[ ]:


del y


# In[ ]:


#Train-test split
import numpy as np
indices = np.random.permutation(9452)

X_train, X_test = X[indices[:7561]], X[indices[7561:]]

y_train, y_test = b[indices[:7561]], b[indices[7561:]]


# In[ ]:


#Model
base_model = tf.keras.applications.InceptionV3(input_shape=(160,160,3), include_top=False, weights='imagenet')
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

#Fine-tune from this layer onwards
fine_tune_at = 200


#Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
    
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(4, activation = 'softmax')
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.save('saved_model/my_model') 


# In[ ]:


model.fit(X_train, y_train, batch_size=64, epochs=4)


# In[ ]:


yHat_test = model.predict(X_test)
y2=(np.argmax(yHat_test, axis=1))
y1=(np.argmax(y_test, axis=1))
from sklearn.metrics import accuracy_score
print(accuracy_score(y1, y2))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y1, y2)

