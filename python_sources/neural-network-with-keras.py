#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load Data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("../input/train.csv")
test = (pd.read_csv("../input/test.csv").values).astype('float32')


# In[ ]:


train.head()


# In[ ]:


train_img = (train.ix[:,1:].values).astype('float32')
labels = train.ix[:,0].values.astype('int32')


# In[ ]:


from keras.utils.np_utils import to_categorical
labels = to_categorical(labels)


# In[ ]:


# Build Model

from keras.models import Sequential
from keras.layers import Dense , Dropout

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(28 *28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[ ]:


# Compile Model

from keras.optimizers import Adam
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Train Model

model_fit = model.fit(train_img, labels, validation_split = 0.05, epochs=24, batch_size=64)


# In[ ]:


# Display Loss and Accuracy

history = model_fit.history
history.keys()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loss_values = history['loss']
val_loss_values = history['val_loss']

epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[ ]:


plt.clf()   # clear figure
acc_values = history['acc']
val_acc_values = history['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# In[ ]:


# Predict

pred = model.predict_classes(test)


# In[ ]:


# To CSV

result = pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),"Label": pred})
result.to_csv("output.csv", index=False, header=True)

