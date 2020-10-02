#!/usr/bin/env python
# coding: utf-8

# <b>A pulsar is a highly magnetized rotating neutron star or white dwarf that emits a beam of electromagnetic radiation. This radiation can be observed only when the beam of emission is pointing toward Earth (much like the way a lighthouse can be seen only when the light is pointed in the direction of an observer), and is responsible for the pulsed appearance of emission.</b>
# 
# <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/PIA18848-PSRB1509-58-ChandraXRay-WiseIR-20141023.jpg/223px-PIA18848-PSRB1509-58-ChandraXRay-WiseIR-20141023.jpg'></img>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <center><font size=4px>A look at the data</font></center>

# In[ ]:


data=pd.read_csv('../input/pulsar_stars.csv')
data.sample(5)


# <center><font size=4px>Correlation of columns</font></center>

# In[ ]:


cols=data.columns
for col in cols:
    print('Column:' + str(col))
    print(np.corrcoef(data['target_class'],data[col]))


# <center><font size=4px>Encoding the target class</font></center>

# In[ ]:


d=pd.get_dummies(data['target_class'])
data=pd.concat([data,d],axis=1)
data.drop('target_class',axis=1,inplace=True)
data.head()


# <center><font size=4px>Making test-set</font></center>

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.drop([0,1],axis=1),data[[0,1]],test_size=0.2)


# <center><font size=4px>Neural network construction</font></center>

# In[ ]:


get_ipython().system('pip install keras-metrics')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam
import keras_metrics as km
model=Sequential()
for i in range(0,4):
    model.add(Dense(100,input_shape=(8,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,input_shape=(8,),activation='relu'))
model.add(Dense(64,input_shape=(8,),activation='relu'))
model.add(Dense(64,input_shape=(8,),activation='relu'))
model.add(Dense(64,input_shape=(8,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,input_shape=(8,),activation='relu'))
model.add(Dense(32,input_shape=(8,),activation='relu'))
model.add(Dense(32,input_shape=(8,),activation='relu'))
model.add(Dense(32,input_shape=(8,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
print('Model summary: ')
model.summary()
print('')
opt = adam(lr=0.001, decay=1e-6)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy',km.precision(),km.recall()])
print('Model fitting...')
model.fit(x_train,y_train, epochs=20, batch_size=32)
score=model.evaluate(x_test,y_test)
print('Accuracy(on Test-data): ' + str(score[1]))


# <center><font size=4px>We have also achieved high precision & recall values. Since the data was heavily skewed, we can use F1 score = (2*P*R)/(P+R) for accurate measurement of our model's performance! <br>Suggestions & feedback always welcomed.</font></center>
