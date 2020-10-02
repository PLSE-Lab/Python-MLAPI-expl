#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # Loading Dataset

# In[ ]:


dataset=pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


dataset


# # Splitting Data into X_train and Y_train

# In[ ]:


X_train=np.array(dataset.drop(['label'],axis=1))
Y_train=np.array(dataset['label'])


# In[ ]:


X_train


# In[ ]:


Y_train


# # Applying Normalization

# In[ ]:


X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_train=X_train.astype('float32')
X_train=X_train/255


# In[ ]:


plt.figure()
plt.imshow(X_train[0][:,:,0])


# # Data Augmentation

# In[ ]:


# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# # Building Model

# In[ ]:


import tensorflow as tf


# In[ ]:


X_train.shape


# In[ ]:


# model=tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu,input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2,2),strides=2),
#     tf.keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu),
#     tf.keras.layers.MaxPooling2D((2,2),strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation=tf.nn.relu),
#     tf.keras.layers.Dense(10,activation=tf.nn.softmax)
# ])
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2,2),strides=2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
             )


# # Training the model

# In[ ]:


model.fit_generator(datagen.flow(X_train,Y_train, batch_size=64),
        epochs = 45, steps_per_epoch = X_train.shape[0]//64)


# # Using trained model to predict test dataset

# In[ ]:


test_df=pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


X_test=np.array(test_df)


# In[ ]:


X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[ ]:


X_test=X_test.astype('float32')


# In[ ]:


X_test/=255


# In[ ]:


predictions=model.predict(X_test)


# In[ ]:


predictions[0]


# In[ ]:


np.argmax(predictions[10])


# In[ ]:


plt.figure()
plt.imshow(X_test[10][:,:,0])


# In[ ]:


X_test.shape


# In[ ]:


predictions.shape


# In[ ]:


results=[]


# In[ ]:


for i in range(28000):
    results.append(np.argmax(predictions[i]))
    


# In[ ]:


results[4]


# In[ ]:


results=pd.Series(results,name="Label")


# In[ ]:


results


# In[ ]:


submission=pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)


# In[ ]:


submission


# # Making Submission

# In[ ]:


submission.to_csv('My_submissions6',index=False)


# In[ ]:


my_sub=pd.read_csv('My_submissions6')


# In[ ]:


my_sub


# In[ ]:




