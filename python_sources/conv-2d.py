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
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dense,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization
from keras.optimizers import RMSprop


# In[ ]:


train_data=pd.read_csv('../input/digit-recognizer/train.csv')
test_data=pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


y_train=train_data['label']


# In[ ]:


X_train=train_data.drop(labels=['label'],axis=1)


# In[ ]:


sns.countplot(y_train)


# In[ ]:


# normalisation
X_train=X_train/255
test_data=test_data/255


# In[ ]:


#Reshape image in 3 dimensions
X_train=X_train.values.reshape(-1,28,28,1)
test_data=test_data.values.reshape(-1,28,28,1)


# In[ ]:


#label encoder
y_train = to_categorical(y_train, num_classes = 10)


# In[ ]:


# train test split data
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.1,random_state=2)


# In[ ]:


plt.imshow(X_train[3][:,:,0])


# In[ ]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size=86
epochs=10


# In[ ]:


model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
           validation_data = (X_test, y_test), verbose = 2)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


Y_pred = model.predict(X_test)

Y_pred_classes = np.argmax(y_test,axis = 1) 

Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10))


# In[ ]:


results = model.predict(test_data)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:




