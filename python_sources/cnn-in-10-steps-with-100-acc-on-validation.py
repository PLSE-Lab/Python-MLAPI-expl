#!/usr/bin/env python
# coding: utf-8

# **Suggestions are welcomed !!**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output
seed=5


# **1. Reading the data**

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.label.value_counts().plot.bar()


# In[ ]:


train.label.value_counts()


# **2. Sampling the data to get equal number of samples of each class**

# In[ ]:


df=train.groupby('label').apply(lambda x: x.sample(3795)).reset_index(drop=True)
df.label.value_counts()


# **3. Shuffling the data**

# In[ ]:


df=df.sample(frac=1,random_state=seed)


# **4. Splitting the data**

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

target=df['label']
X=df.drop('label',axis=1)
target=to_categorical(target)

train_X,val_X,train_y,val_y=train_test_split(X,target,test_size=0.01,random_state=seed)


# **5. Normalizing and Reshaping the data**

# Data is normalized because the optimizers converge faster when the values are between (0,1) rather than (0,255).
# Width and height of images is 28 pixels. 

# In[ ]:


train_X=train_X/255
val_X=val_X/255

train_X=train_X.values.reshape(-1,28,28,1)
val_X=val_X.values.reshape(-1,28,28,1)


# **6. Plotting one of the samples**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


plt.imshow(train_X[0][:,:,0])


# **7. Data Augmentation**

# We use data augmentation to generate more samples of each class and to make our model more robust.
# 
# 1. We shift width of samples randomly by 20 percent.
# 2. We shift height of samples randomly by 20 percent.
# 3. We rotate the samples randomly by 10 degrees.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


train_datagen=ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,
                                rotation_range=10)

train_datagen.fit(train_X)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,BatchNormalization,Dropout,Flatten,Dense
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# **8. Setting up the model**

# In[ ]:


model=Sequential()
model.add(Conv2D(64,3,input_shape=(28,28,1),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Conv2D(128,3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Conv2D(256,3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Conv2D(512,3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Conv2D(1024,3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Conv2D(1024,3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,padding='same'))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# **9. Using callbacks to avoid overfitting**

# In[ ]:


est=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
rlp=ReduceLROnPlateau(monitor='val_loss',patience=10,factor=0.1,min_delta=0.0001)
call_backs=[est,rlp]


# In[ ]:


#tune epochs to 30 for 100 percent accuracy on validation

batch_size=64
result=model.fit_generator(train_datagen.flow(train_X,train_y,batch_size=batch_size),epochs=1,
                  callbacks=call_backs,validation_data=(val_X,val_y),validation_steps=1,
                          steps_per_epoch=train_X.shape[0]//batch_size)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(result.history['loss'],label='Train loss')
plt.plot(result.history['val_loss'],label='Validation loss')
plt.legend(loc='best')


# **10. Confusion matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix

val_pred_probs=model.predict(val_X)
val_labels=np.argmax(val_pred_probs,axis=1)
true_labels=np.argmax(val_y,axis=1)
cm=confusion_matrix(val_labels,true_labels)

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)


# Not even one sample is misclassified! But this can also be attributed to the fact that validation set size is very small. Nevertheless, I am amazed :D

# In[ ]:


test=test/255
test=test.values.reshape(-1,28,28,1)


# In[ ]:


test_probs=model.predict(test)
test_labels=np.argmax(test_probs,axis=1)


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')
sub['label']=test_labels
sub.to_csv('Submission.csv',index=False)

