#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/cars-train.csv')
val = pd.read_csv('../input/cars-test.csv')

x_train = pd.get_dummies(train.drop(['class', 'car.id'], axis=1))
x_val = pd.get_dummies(val.drop(['class', 'car.id'], axis=1))
y_train = pd.get_dummies(train['class'])
y_val = pd.get_dummies(val['class'])


# We're going to jump straight into overkill with this Keras model.  The activation function being used in scaled exponential linear unit and it's cool because it causes NN's to self-normalize.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, PReLU, BatchNormalization

model = Sequential()
model.add(Dense(128, input_dim=(21), activation = 'selu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'selu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'selu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'selu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(4, activation = 'softmax'))

model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


# Again, we're just going to say screw it and run a lot of epochs on a very small batch size.  This dataset is very simple so, again, I expect we can get nearly perfect results with it.

# In[ ]:


model.fit(x_train, y_train, epochs = 600, batch_size = 32)


# In[ ]:


score = model.evaluate(x_train, y_train)

print('\ntrain loss is: ' + str(score[0].round(4)))
print('train accuracy is: ' + str(score[1]))

score = model.evaluate(x_val, y_val)

print('\ntest loss is: ' + str(score[0].round(4)))
print('test accuracy is: ' + str(score[1]))


# The above model is definitely overkill but it has perfect *accuracy* on both training and validation sets, .0003 loss on the training set and .0018 loss on the validation.  That's good enough for government work to me.  Now to predict the classes of our test set and convert them from the numbers keras uses back to classes.

# In[ ]:


test = pd.read_csv('../input/cars-final-prediction.csv')
test_new = pd.get_dummies(test.drop(['car.id'], axis=1))
preds = model.predict_classes(test_new)
keras_dict = {0: 'acc', 1: 'good', 2: 'unacc', 3: 'vgood'}
converted_preds = []
for prediction in preds:
    converted_preds.append(keras_dict[prediction])
test['class']=converted_preds
output_csv = test[['car.id', 'class']]
output_csv


# In[ ]:


output_csv.to_csv('car-submission.csv', index=False)


# In[ ]:




