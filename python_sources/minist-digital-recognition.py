#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()


# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')
test.head()


# In[ ]:


test.shape


# In[ ]:



x, y = train.iloc[:,1:785], train.iloc[:,0]
print(x.shape)
print(y.shape)


# Standardardize The Training Input

# In[ ]:


x_standard = []
for i in range(0,x.shape[0]):
    x_standard.append(x.iloc[i,:].values.reshape(28,28)/255)


# Convert Into Array

# In[ ]:


x_standard_array = np.array(x_standard)


# In[ ]:


x_standard_array.shape


# Show some standardized digit's image
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt



# In[ ]:


plt.imshow(x_standard_array[8])
plt.show()
print(y[8])


# Implement Small Convnet Model

# In[ ]:


from keras import layers
from keras import models
from keras.utils import to_categorical


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add((layers.Conv2D(64,(3,3), activation = 'relu')))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))


# In[ ]:


model.compile(optimizer = 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])


# In[ ]:


train_images = x_standard_array.reshape((42000,28,28,1))
y_labels = to_categorical(y)


# In[ ]:


model.fit(train_images, y_labels, epochs = 5)


# In[ ]:


test_images = test.values.reshape(28000,28,28)/255
test_images = np.array(test_images)
test_images.reshape((28000,28,28,1))
test_images.shape


# In[ ]:


plt.imshow(test_images[1])
plt.show()


# In[ ]:


test_images[0].shape


# In[ ]:


pred=[]
for x in test_images:
    pred.append(model.predict(x.reshape(1,28,28,1)))


# In[ ]:


eval_results = []
for i in range(0,len(pred)):
    eval_results.append(int(np.argmax(pred[i],axis=1)))


# In[ ]:


eval_results[0:10]


# **Submission**

# In[ ]:


results = pd.Series(eval_results, name = 'Label')

submission = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), results], axis = 1)


# In[ ]:


submission.to_csv('Deep_Learning_Results.csv',index = False)

