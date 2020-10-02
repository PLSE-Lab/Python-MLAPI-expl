#!/usr/bin/env python
# coding: utf-8

# # Problem Statement and Business Case
# * The Dataset Contains series of images that can be used to solve the Happy House problem!
# * We need ti build an artificial neural network that can detect smiling faces.
# * Only smiling People allowed to enter the house!
# * The train set has 600 examples. The test has 150 examples.
# 
# 
# 
# 

# # 1 - Importing Libraries

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import h5py

get_ipython().run_line_magic('matplotlib', 'inline')
# http://setosa.io/ev/image-kernels/


# # 2 - Importing Data

# def load_dataset(path_to_train, path_to_test):
#     train_dataset = h5py.File(path_to_train)
#     train_x = np.array(train_dataset['train_set_x'][:])
#     train_y = np.array(train_dataset['train_set_y'][:])
# 
#     test_dataset = h5py.File(path_to_test)
#     test_x = np.array(test_dataset['test_set_x'][:])
#     test_y = np.array(test_dataset['test_set_y'][:])
# 
#     # y reshaped
#     train_y = train_y.reshape((1, train_x.shape[0]))
#     test_y = test_y.reshape((1, test_y.shape[0]))
# 
#     return train_x, train_y, test_x, test_y
# X_train, y_train, X_test, y_test=load_dataset('../input/train_happy.h5','../input/test_happy.h5')

# In[24]:


t = h5py.File('../input/train_happy.h5')
for key in t.keys():
    print(key)


# In[25]:


s = h5py.File('../input/test_happy.h5')
for key in s.keys():
    print(key)


# In[26]:


happy_trainging = h5py.File('../input/train_happy.h5')
happy_testing = h5py.File('../input/test_happy.h5')


# In[27]:


X_train = np.array(happy_trainging['train_set_x'][:])
y_train = np.array(happy_trainging['train_set_y'][:])

X_test = np.array(happy_testing['test_set_x'][:])
y_test = np.array(happy_testing['test_set_y'][:])


# In[28]:


print('X_train.shape = ',X_train.shape)
print('X_test.shape = ',X_test.shape)


# In[29]:


y_train # target class


# In[30]:


y_test.shape


# In[31]:


y_test


# # 3 - Visualization of the Data[](http://)set

# In[32]:


index = random.randint(1,600)
plt.imshow(X_train[index])
print('index = ', index,', result = ', y_train[index])


# In[33]:


W_grid = 5
Len_grid = 5

n_training =len(X_train)

fig,axes=plt.subplots(Len_grid,W_grid,figsize = (25,25))
axes = axes.ravel()

for i in np.arange(0, W_grid * Len_grid ):
    index = np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index],fontsize = 25)
    axes[i].axis('off')
    


# # 4 - Training the Model

# In[34]:


#Normalization
X_train = X_train / 255
X_test = X_test / 255


# In[35]:


plt.imshow(X_train[1])


# In[36]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[37]:


cnn_model = Sequential()
cnn_model.add(Conv2D(64, 6, 6,input_shape = (64, 64, 3),activation = 'relu' ))
cnn_model.add(MaxPool2D(pool_size =(2, 2)))

cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(64, 5, 5, activation = 'relu'))
cnn_model.add(MaxPool2D(pool_size =(2, 2)))


cnn_model.add(Flatten())

cnn_model.add(Dense(output_dim = 128, activation = 'relu'))

cnn_model.add(Dense(output_dim = 1, activation = 'sigmoid'))


# In[38]:


cnn_model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.001), metrics = ['accuracy'])


# In[39]:


epochs = 50

history =cnn_model.fit(X_train, y_train, batch_size = 30, nb_epoch = epochs, verbose = 1)


# # 5 - Evaluating the Model

# In[40]:


predict_classes = cnn_model.predict_classes(X_test)


# In[41]:


predict_classes


# In[42]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,predict_classes)
sns.heatmap(cm, annot = True)


# In[43]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predict_classes ))


# In[44]:


L = 10
W = 15
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    if (predict_classes[i] != y_test[i]):
        axes[i].imshow(X_test[i])
        axes[i].set_title('prediction class ={}\n True Class = {}'.format(predict_classes[i],y_test[i]))
        axes[i].axis('off')
    
plt.subplots_adjust(wspace = 0.5)


# In[ ]:




