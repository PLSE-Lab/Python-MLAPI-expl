#!/usr/bin/env python
# coding: utf-8

# ## Let's GO!!!
# 
# In this Kernel We will be using Convolution Neural Networks:
# - [Understanding the data](#1)
# - [Preparing the data](#2)
# - [Building a CNN](#3)
# - [How to make submission](#4)
# 
# 
# <p><font size='4' color='green'> If you like this kernel then please consider giving an upvote !</font></p>

# ## Importing necessary libraries !

# In[ ]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau


# ### Loading the data

# In[ ]:


train=pd.read_csv('../input/Kannada-MNIST/train.csv')
test=pd.read_csv('../input/Kannada-MNIST/test.csv')
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# ### Understanding the data <a id="1" ></a>

# In[ ]:


print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print('The Test  dataset has {} rows and {} columns'.format(test.shape[0],test.shape[1]))


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)
test=test.drop('id',axis=1)


# ### Checking wether target column is balanced or not
# * Target variable **label** is balanced

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='label',data=train,palette='RdBu_r')


# ## Data preparation <a id='2'></a>

# In[ ]:


X_train=train.drop('label',axis=1)
Y_train=train.label


# In[ ]:


X_train=X_train/255
test=test/255


# ### Reshape

# In[ ]:


X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# In[ ]:


print('The shape of train set now is',X_train.shape)
print('The shape of test set now is',test.shape)


# ### Encoding Target Values

# Now we will encode our target value.Keras inbuild library to_categorical() is used to do the on-hot encoding.

# In[ ]:


Y_train=to_categorical(Y_train)


# ### Splitting train and test

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)


# In[ ]:


plt.imshow(X_train[0][:,:,0])


# It's Nine in Kannada
# 

# ### Data Augmentation!

# In[ ]:


# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


datagen.fit(X_train)


# ## Modelling <a id='3' ></a>

# In[ ]:



model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


epochs=5 #change this to 30 if you need to get better score
batch_size=64


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# ## Submission <a id='4'></a>
# 

# In[ ]:


test=pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


test_id=test.id

test=test.drop('id',axis=1)
test=test/255
test=test.values.reshape(-1,28,28,1)


# In[ ]:


test.shape


# We will make our prediction using our CNN model.

# In[ ]:


y_pre=model.predict(test)     ##making prediction
y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels


# In[ ]:


sample_sub['label']=y_pre
sample_sub.to_csv('submission.csv',index=False)


# In[ ]:


sample_sub.head()

