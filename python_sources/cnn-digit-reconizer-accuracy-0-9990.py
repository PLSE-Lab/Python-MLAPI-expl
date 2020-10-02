#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


target = train['label']
train = train.drop('label', axis = 1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)


# In[ ]:


train_reshape = train.values.reshape(42000,28,28,1)


# In[ ]:


train_reshape = train_reshape/255


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[ ]:


# Initialising CNN
classifier = Sequential()

#Adding first conv layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (28,28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adiing second conv layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

#Full connection
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


# training ANN on train dataset
classifier.fit(train_reshape, target, batch_size = 20 , epochs = 15)


# In[ ]:


print(test.shape)


# In[ ]:


test_reshape = test.values.reshape(28000, 28,28,1)


# In[ ]:


test_pred = classifier.predict(test_reshape)


# In[ ]:


pred_int = []
for i in range(test_pred.shape[0]):
    pred_ratios = test_pred[i]
    for j in range(10):
        if(pred_ratios[j] == max(pred_ratios)):
            number = int(j)
    pred_int = np.append(pred_int, number)
pred_int = pred_int.astype(int)


# In[ ]:


submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = pred_int
submission


# In[ ]:


submission.to_csv('submission.csv', index = False)

