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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical # to convert one hot encoding
from keras.optimizers import Adam
import os
print(os.listdir("../input"))
sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


Y_train = train_data['label']
#Y_train.value_counts().plot.bar()
sns.countplot(Y_train)
X_train = train_data.drop('label',axis = 1)
del train_data


# In[ ]:


X_train.isnull().any().describe()


# In[ ]:


test_data.isnull().any().describe()


# In[ ]:


X_train = X_train/255.0
test_data = test_data/255.0
type(X_train)


# In[ ]:


#reshape
X_train = X_train.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
type(X_train)


# In[ ]:


Y_train = to_categorical(Y_train,num_classes = 10)


# In[ ]:


#spliting training and testing data
random_seed = 2
X_train,test_X,Y_train,test_Y = train_test_split(X_train,Y_train,test_size = 0.1,random_state = random_seed)
type(X_train)


# In[ ]:


g = plt.imshow(X_train[1][:,:,0])


# In[ ]:


g = plt.imshow(X_train[4][:,:,0])


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu',padding = 'same'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'))

model.add(Flatten())
model.add(Dense(64,activation = 'relu'))

model.add(Dense(10,activation = 'softmax'))


# In[ ]:


model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


history = model.fit(x = X_train,y = Y_train,batch_size = 100,epochs = 8,validation_data=(test_X,test_Y))


# In[ ]:


plt.plot(history.history['val_loss'], color='r', label= "validation loss")
plt.title("Test Loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


result = model.predict(test_data)


# In[ ]:


results = np.argmax(result,axis = 1)
results


# In[ ]:


plt.plot(history.history['val_accuracy'], color='b', label= "validation accuracy")
plt.title("Test Accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


Label = pd.Series(results,name = 'Label')
ImageId = pd.Series(range(1,28001),name = 'ImageId')
submission = pd.concat([ImageId,Label],axis = 1)
submission.to_csv('submission.csv',index = False)
submission


# In[ ]:


ImageId.head()

