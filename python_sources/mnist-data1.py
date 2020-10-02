#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras 


# In[ ]:


print ('we are using tensorflow version ' , tf.__version__)


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


test =  pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


X = train.drop(columns=['label'])


# In[ ]:


y = train['label']


# In[ ]:


y.shape


# In[ ]:


print ('shape of train set ; ', train.shape)
print ('shape of traing image set ; ', X.shape)
print ('shape of training label set ; ', y.shape)
print ('shape of test set ; ', test.shape)


# In[ ]:


print ('Nan Values in training set', train.isna().sum().sum())
print ('Nan Values in test set', test.isna().sum().sum())


# In[ ]:


sns.countplot(y)


# In[ ]:


X.info()


# In[ ]:


# we need our data to present in as float32
# converting uint8 to float32
train =np.array(train).astype('float32')
test = np.array(test).astype('float32')
X =np.array(X).astype('float32')
y = np.array(y).astype('float32')


# In[ ]:


print ('dtype of train is :' , train.dtype)
print ('dtype of test is :' , test.dtype)
print ('dtype of X is :' , X.dtype)
print ('dtype of y is :' , y.dtype)


# In[ ]:


y.shape


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure (figsize=(25,10))
x,q = 10,4
for i in range(10):
    plt.subplot(q,x,i+1)
    plt.imshow(X[i].reshape((28,28)))
    plt.xlabel(y[i])
    
    


# In[ ]:


y.shape


# In[ ]:


y = tf.keras.utils.to_categorical(y)


# In[ ]:


from sklearn.preprocessing import Normalizer

transformer = Normalizer()
X = transformer.fit_transform(X)
test = transformer.transform(test)


# In[ ]:


# Initialize Sequential model
model = tf.keras.models.Sequential() # Instantiating keras sequential models from keras 
model.add(tf.keras.layers.BatchNormalization())
# First layer (input layer) of  28*28 = 784 after flattening the image of 28 * 28 picxels
model.add(tf.keras.layers.Dense(784,input_dim=784,kernel_initializer='uniform', activation='relu'))
# second layer 
model.add(tf.keras.layers.Dense(392, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
# third layer
model.add(tf.keras.layers.Dense(181, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
# Final layer with activation function as softmax and 10 neurons 
model.add(tf.keras.layers.Dense(10, activation='softmax'))



# Create optimizer with non-default learning rate
#sgd_optimizer = tf.keras.optimizers.SGD(lr=0.2)

# Compile the model
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X,y,epochs=50,batch_size=32)


# Now we will create a model with CNN Network 

# In[ ]:


tf.keras.backend.clear_session()
#Initialize model, reshape & normalize data
model2 = tf.keras.models.Sequential()

#Reshape data from 2D (28,28) to 3D (28, 28, 1)
model2.add(tf.keras.layers.Reshape((28,28,1),input_shape=(784,)))
#normalize data
model2.add(tf.keras.layers.BatchNormalization())


# In[ ]:


#Add first convolutional layer
model2.add(tf.keras.layers.Conv2D(32, #Number of filters 
                                 kernel_size=(3,3), #Size of the filter
                                 activation='relu'))

#Add second convolutional layer
model2.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))

#Add MaxPooling layer
model2.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())

#Flatten the output
model2.add(tf.keras.layers.Flatten())

#Dense layer
model2.add(tf.keras.layers.Dense(128, activation='relu'))


# In[ ]:


#Output layer
model2.add(tf.keras.layers.Dense(10, activation='softmax'))


model2.compile(optimizer='adam', 
              loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model2.summary()


# In[ ]:


#Train the model
model2.fit(X,y,epochs=10,batch_size=32)


# In[ ]:


predict = model.predict(test)
predict2 = model2.predict(test)


# In[ ]:


label = np.argmax(predict,axis=1)
label2 = np.argmax(predict2,axis=1)


# In[ ]:


df = pd.DataFrame(label)


# In[ ]:


df.columns= ['label']


# In[ ]:


df['ImageId'] =  np.arange(1,len(label)+1)


# In[ ]:


df = df.set_index(df['ImageId'])


# In[ ]:


df_check = df.drop(columns=['ImageId'])


# In[ ]:


df_check


# In[ ]:


df_check.to_csv(r'check_result2.csv',index=False)


# In[ ]:


my_submission2 = pd.DataFrame({'ImageId': np.arange(1,len(label)+1), 'Label': label2})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)
my_submission2.to_csv('submission2.csv', index=False)


# In[ ]:


my_submission


# In[ ]:


label


# In[ ]:


my_submission = pd.DataFrame({'ImageId': np.arange(1,len(label)+1), 'Label': label})


# In[ ]:


my_submission


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:




