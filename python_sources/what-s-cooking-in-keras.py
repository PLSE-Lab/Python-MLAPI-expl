#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


len(train_df['cuisine'].unique())


# In[ ]:


train_df['cuisine'].value_counts().plot(kind='barh')


# In[ ]:


target=train_df['cuisine']
train=train_df.drop('cuisine',axis=1)
test=test_df
target.head()


# In[ ]:


t=Tokenizer()
t.fit_on_texts(train['ingredients'])
train_encoded=t.texts_to_matrix(train['ingredients'],mode='tfidf')


# In[ ]:


cuisines=train_df['cuisine'].unique()
label2index={cuisine:i for i,cuisine in enumerate(cuisines)}
y=[]

for item in target:
    if item in label2index.keys():
        y.append(label2index[item])
y_encoded=to_categorical(y,20)


# In[ ]:


print(train_encoded.shape)
print(y_encoded.shape)


# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(256,input_shape=[train_encoded.shape[1], ],activation='relu',name='hidden_1'))
    model.add(Dropout(0.4, name='dropout_1'))
    
    #model.add(Dense(64,activation='relu',name='hidden_2'))
    #model.add(Dropout(0.2,name='dropout_2'))
    
    model.add(Dense(20,name='output'))
    
    model.compile(optimizer='adam',
                  loss='categorical_hinge',
                  metrics=['accuracy']
                )
    
    return model


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(train_encoded,y_encoded,test_size=0.2,random_state=22)


# In[ ]:


model=build_model()
model.summary()


# In[ ]:


monitor=[
    EarlyStopping(monitor='val_loss',patience=5,verbose=1),
    ModelCheckpoint('best-model-0.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
]

model.fit(X_train,y_train,
         validation_data=(X_val,y_val),
         epochs=100,
         callbacks=monitor,
         batch_size=128)


# In[ ]:


test_encoded=t.texts_to_matrix(test_df['ingredients'],mode='tfidf')
test_encoded.shape


# In[ ]:


model.load_weights('best-model-0.h5')
y_pred=model.predict(test_encoded).argmax(axis=1)

results=[]

for i in y_pred:
    for k,v in label2index.items():
        if v==i:
            results.append(k)

results[:10]


# In[ ]:


submission=pd.DataFrame(list(zip(test_df['id'],results)),columns=['id','cuisine'])
submission.to_csv('submission.csv',header=True,index=False)


# In[ ]:


submission=pd.read_csv('submission.csv')
submission.head()


# In[ ]:




