#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import keras
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.optimizers import SGD


# In[ ]:


train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


train_data=train_data.sample(frac=1).reset_index(drop=True)


# In[ ]:


labels=train_data['label']


# In[ ]:


train_data=train_data.drop(['label'],axis=1)


# In[ ]:


labels.value_counts()


# In[ ]:


labels = keras.utils.to_categorical(labels,len(labels.value_counts()))


# In[ ]:


train_data=train_data.astype('float32')/255


# In[ ]:


train_X=train_data[:36000]
test_X=train_data[36000:]
train_y=labels[:36000]
test_y=labels[36000:]


# In[ ]:


train_X=train_X.values.reshape(train_X.shape[0],28,28,1)
test_X=test_X.values.reshape(test_X.shape[0],28,28,1)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


model.fit(train_X,train_y,batch_size=32,epochs=35,verbose=1,validation_data=(test_X, test_y))


# In[ ]:


test_data=test_data.astype('float32')/255


# In[ ]:


test=test_data.values.reshape(len(test_data),28,28,1)


# In[ ]:


k=model.predict_classes(test)
res=pd.DataFrame(data=k)
res.index+=1
res['ImageId']=res.index
res=res.rename(index=str, columns={0: "Label1"})
res['Label']=res['Label1']
res=res.drop(['Label1'],axis=1)


# In[ ]:


res.to_csv('result.csv', index=False)

