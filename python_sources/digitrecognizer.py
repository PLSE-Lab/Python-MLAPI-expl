#!/usr/bin/env python
# coding: utf-8

# **Hey there!**
# 
# 
# > This notebook is an introductory exercise to implement a Digit Recognizer using Convolutional Neural Networks using Keras API. The visualizations are done using Matplotlib.
# In case of any errors, do comment below!. Suggestions are always welcome.
# 
# 

# In[ ]:


import numpy as np, pandas as pd
import random

import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Input, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"]= [5,10]


# The data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', low_memory=True)\ntest = pd.read_csv('../input/test.csv', low_memory=True)")


# In[ ]:


print(train.shape, test.shape) 
## here each image is 28x28 matrix, train contains an extra (first)column 'label' containing actual 
## decimal representation of the image.


# In[ ]:


train.columns


# In[ ]:


train.loc[:,'label'].value_counts()


# **Let's visualize a random image from test and train data_set**

# In[ ]:


rows = 2; columns = 5
fig = plt.figure(figsize=(14, 8))
for i in range(1, columns*rows+1):
    image = np.array(train.iloc[i,1:]).reshape((28,28))
    label = train.loc[i,'label']
    ax = fig.add_subplot(rows,columns,i)
    ax.imshow(image, cmap='gray')
    ax.set_title(label)    


# **Let's build a CNN model using keras API**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size=0.33, random_state=42, shuffle = True)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


X_train = np.array(X_train).reshape((X_train.shape[0],28,28,1)).astype('float32')
X_test = np.array(X_test).reshape((X_test.shape[0],28,28,1)).astype('float32')


# In[ ]:


## Normalization
X_train /= 255
X_test /= 255


# Class Distribution in validation/train data

# In[ ]:


print(y_train.value_counts(), y_test.value_counts())


# In[ ]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 


# In[ ]:


y_test.shape


# **Learning Rate Scheduler**

# In[ ]:


def learning_rate_scheduler(epoch):
    if epoch <5:
        lr = 1e-3
    if epoch >= 5 and epoch <=20:  lr = 3e-4
    if epoch >20: lr= 1e-5
    return lr


# **Building and Training CNN Model using Sequential API.**

# In[ ]:


model = Sequential()

model.add(Conv2D(input_shape = (28,28,1),data_format='channels_last', filters = 128, kernel_size=(5,5), activation ='relu'))
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation = 'relu'))
model.add(Conv2D(filters = 28, kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = Adam(lr = 0.01),loss = 'categorical_crossentropy', metrics = ['accuracy'])

epoch = 6
batch_size = 64
change_lr = LearningRateScheduler(learning_rate_scheduler)
history = model.fit(X_train, y_train, epochs=epoch, batch_size = batch_size, validation_data=[X_test,y_test], 
                    callbacks=[change_lr], shuffle = True)


# In[ ]:


loss = history.history['loss']
acc = history.history['acc']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']


# In[ ]:


row,col = (1,2)

fig = plt.figure(figsize = (10,5))

ax1 = fig.add_subplot(121)
ax1.plot(loss, color='red',label='train_loss')
ax1.set_title('loss')
ax1.plot(val_loss, color='green',label= 'val_loss')
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(122)
ax2.plot(acc, color='red', label = 'train accuracy')
ax2.plot(val_acc, color='green', label='val_accuracy')
ax2.set_title('accuracy')
ax2.legend(loc='lower right')

plt.plot()
plt.tight_layout()


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:





# **Now submission for test data**

# In[ ]:


test = np.array(test)
test = test.reshape((test.shape[0],28,28,1))/255


# In[ ]:


out = model.predict(test)


# In[ ]:


i = random.randint(1,28000)
image = np.array(test[i]).reshape((28,28))
plt.imshow(image, cmap='gray')
plt.title(f'Predictions for index {i}: {np.argmax(out[i])}')


# In[ ]:


output = np.argmax(out,axis =1)


# In[ ]:


pd.DataFrame(output, columns =['output'])['output'].value_counts()
                                                      


# In[ ]:


ImageId = list(range(1,28001))


# In[ ]:


submission = pd.DataFrame({'ImageId':ImageId, 'Label':output})


# In[ ]:


submission.to_csv('submission.csv',index = 'False')


# **A basic CNN model works unsurprisingly well with on validation data.**

# There is a lot of tweaks could be done w.r.t.hyperparameters, such as batch_size, learning rate, epochs, trying out different optimizers such as Adam, RMSProp, Adagrad,Adadelta. 
# The obtained accuracy scores could be further elevated.
