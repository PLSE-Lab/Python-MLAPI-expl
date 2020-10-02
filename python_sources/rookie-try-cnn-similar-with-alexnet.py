#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import OneHotEncoder


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


y_train = train['label']
x_train = train.drop('label', axis=1)
y_train.shape, x_train.shape


# In[ ]:


x_train.head(5)


# In[ ]:


y_train.value_counts()


# In[ ]:


x_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


#  **Normalization**

# In[ ]:


x_train = x_train/255.0
test = test/255.0


# **Reshape**
# * [np.reshape](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.reshape.html) detail
# * x_train is numpy.ndarray format now

# In[ ]:


x_train = x_train.values.reshape(-1,28,28,1)


# In[ ]:


test = test.values.reshape(-1,28,28,1)


# In[ ]:


test.shape


# **Label Encoding**
# * Use the Scikit to make one-hot encoding label (Be careful about the format), which must be used with **np.array**.
# * Detail [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

# In[ ]:


# Use the Scikit to make one-hot encoding label
y_train = pd.DataFrame(data=y_train)
one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot.fit(y_train.values)
y_train = one_hot.transform(y_train.values).toarray()


# * Now, y_train becomes** numpy.ndarray**

# In[ ]:


y_train, y_train.shape


# **Splitting dataset as train data and cross-validation data**
# * Detail [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# In[ ]:


from sklearn.model_selection import train_test_split
random_seed = 3
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[ ]:


g = plt.imshow(x_train[0][:,:,0])


# **Build AlexNet with Keras**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential([
    Conv2D(filters = 64, input_shape=(28,28,1), kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='valid'),
    Activation('relu'),
    BatchNormalization(),
    
    Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    BatchNormalization(),

    Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Flatten(),
    Dense(2048),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(2048),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),

    Dense(800),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(10),
    Activation('softmax'),
])


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, batch_size=200, validation_data=(x_val,y_val), epochs = 10)


# In[ ]:


results = model.predict(test)


# In[ ]:


results = np.argmax(results, axis=1)
# select the indix with the maximum probability
results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_submission1.csv",index=False)

