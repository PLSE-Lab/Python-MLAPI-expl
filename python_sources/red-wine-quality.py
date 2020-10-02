#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas
import numpy

from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
data = pandas.read_csv('../input/winequality-red.csv')


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data.sample(30)


# In[ ]:


data.isnull()


# In[ ]:


data = data.values

X = data[:,0:11]
Y = data[:,11]

X.shape


# In[ ]:


from keras.utils import np_utils
Y = np_utils.to_categorical(Y)
Y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1 ,random_state = 0)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.11 ,random_state = 0)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_val.shape


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(40, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
model.add(Dense(9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(9, kernel_initializer='uniform', activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


model.fit(X_train, Y_train, epochs = 100, batch_size = 10)


# In[ ]:


scores = model.evaluate(X_val, Y_val)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


from keras.models import model_from_json

model_json = model.to_json()
with open(r'red_wine_quality_model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights(r'red_wine_quality_model.h5')

