#!/usr/bin/env python
# coding: utf-8

# Loading the input.
# Analyzing the input

# In[16]:



import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[17]:


train=pd.read_csv("../input/train.csv")


# In[18]:


train.shape


# In[19]:


#Split the training dataset into x and y.
y_train=train.iloc[:,0]
x_train=train.iloc[:,1:]


# In[20]:


from keras.utils import np_utils
nb_classes=10
# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)


# In[21]:


del train


# In[22]:


x_train = x_train.astype('float32')
x_train=x_train/255


# In[23]:


from sklearn.model_selection import train_test_split
x_tre, x_val, y_tre, y_val = train_test_split(x_train, Y_train, test_size=0.33, random_state=42)


# In[24]:


print(x_tre.shape)
print(x_val.shape)
print(y_tre.shape)
print(y_val.shape)


# In[25]:


print(type(x_tre))
print(type(x_val))
print(type(y_tre))
print(type(y_val))


# In[26]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=784))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()


# In[31]:


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=20, min_lr=0.001)
history=model.fit(x_tre, y_tre, validation_data=(x_val, y_val), callbacks=[reduce_lr])


# In[ ]:


"""""history=model.fit(x_tre, y_tre, 
                  epochs=50, 
                  batch_size=128,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stop])


# In[32]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[33]:


test=pd.read_csv("../input/test.csv")
test=test/255
test.shape


# In[34]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:





# In[ ]:




