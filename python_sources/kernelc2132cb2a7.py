#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical


# In[ ]:





# In[ ]:


data=pd.read_csv("../input/foresta/forest_data.csv")
teste=pd.read_csv("../input/testef/forest_data_teste.csv")


# In[ ]:


data.head()


# In[ ]:



x=np.array(pd.get_dummies(data.drop("Label",1)))
y=np.array(pd.get_dummies(data.Label))
xtest1=np.array(pd.get_dummies(teste.drop("Label",1)))
ytest1=np.array(pd.get_dummies(teste.Label))


# In[ ]:





# In[ ]:



xtrei,xtest,ytrei,ytest=train_test_split(x,y,test_size=0.30)


# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xtrei, ytrei, epochs=500,batch_size=100,validation_data=(xtest1, ytest1))



# In[ ]:


scores = model.evaluate(xtest, ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


scores = model.evaluate(xtest1, ytest1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:





# In[ ]:





# In[ ]:




