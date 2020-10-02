#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


X = df.drop('Outcome',axis=1)
y = df['Outcome']


# In[ ]:


model = Sequential()
model.add(Dense(256,input_dim=8,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X,y,batch_size=140,epochs=1500)


# In[ ]:


_,accuracy = model.evaluate(X,y,verbose=0)


# In[ ]:


accuracy*100


# In[ ]:


predictions = model.predict_classes(X)


# In[ ]:


predictions


# In[ ]:





# In[ ]:




