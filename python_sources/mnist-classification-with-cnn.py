#!/usr/bin/env python
# coding: utf-8

# # Check Files 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers,models,initializers


# # Set Seeds

# In[ ]:


np.random.seed(20200422)
tf.random.set_seed(20200422)


# # Load Sample Submission and MNIST

# In[ ]:


sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# # Confirm Data 
# Check the initial parts and the shapes of the datasets.

# In[ ]:


sample_submission.head()


# In[ ]:


train.shape


# 'Train' has 42000 rows and 785 columns.
# 
# It means there are 42000 images and each of them has 784 pixel information(1 column is 'Label').

# In[ ]:


train.head()


# In[ ]:


test.shape


# 'Test' has 28000 rows and 784 columns.

# In[ ]:


test.head()


# # Prepare For Using CNN
# Separate images and labels, and normalize pixel value of images.

# In[ ]:


train_labels = tf.keras.utils.to_categorical(train['label'],10)
train.drop('label',axis=1,inplace=True)

train_images = train/255
test_images = test/255


# # Construct Learning Model

# In[ ]:


model = models.Sequential()
model.add(layers.Reshape((28,28,1),input_shape=(28*28,),name='reshape'))
model.add(layers.Conv2D(32,(5,5),padding='same',
                        kernel_initializer=initializers.TruncatedNormal(),
                        use_bias=True,activation='relu',name='conv_filter1'))
model.add(layers.MaxPooling2D((2,2),name='max_pooling1'))
model.add(layers.Conv2D(64,(5,5),padding='same',
                        kernel_initializer=initializers.TruncatedNormal(),
                        use_bias=True,activation='relu',name='conv_filter2'))
model.add(layers.MaxPooling2D((2,2),name='max_pooling2'))
model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(1024,activation='relu',
                      kernel_initializer=initializers.TruncatedNormal(),
                      name='hidden'))
model.add(layers.Dropout(rate=0.5,name='dropout'))
model.add(layers.Dense(10,activation='softmax',name='softmax'))

model.summary()


# # Model Compile and Fit
# Set optimization algorithm and loss function, and run the learning process.

# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
histroy = model.fit(train_images,train_labels,batch_size=128,epochs=10)


# Check the fluctuation of accuracy and loss function.

# In[ ]:


pd.DataFrame({'acc':histroy.history['acc'],'loss':histroy.history['loss']}).plot()


# # Predict With Test Images and Submit

# In[ ]:


predict_value = model.predict(test_images)
predict_index = np.argmax(predict_value,axis=1)

sample_submission['Label'] = predict_index
sample_submission.to_csv('submission.csv',index=False)

