#!/usr/bin/env python
# coding: utf-8

# Hello All !
# 
# Again back with another kaggle attempt. 
# 
# Taken reference from [Cats and Dogs Kernel](https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial)
# 
# Using CNN to resolve this problem. Got a score of 0.14294. However still think rounding result might cause problem. Any suggestions welcome :)
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


X = train[[col for col in train.columns if "stop" in col]]


# In[ ]:


y = train[[col for col in train.columns if "start" in col]]


# In[ ]:


X= np.array(X)
y = np.array(y)


# I have tried with skipping normalization as well adding it didn't make much difference. However normalization is an important step but in case of binary I was'nt sure

# In[ ]:


#X = normalize(X)
#y = normalize(y)


# In[ ]:


X= X.reshape(-1, 20, 20, 1)


# In[ ]:


model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(400, activation='softmax'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X,y, epochs=10, batch_size=32, validation_split=0.1)


# In[ ]:


test.head()


# In[ ]:


x_test = test.drop(['id', 'delta'], axis=1)


# In[ ]:


x_test = np.array(x_test).reshape(-1,20,20,1)


# In[ ]:


# = normalize(x_test)


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


#np.argmax(predictions[0][340])
#int(round(predictions[1][0]))
predictions = predictions.round()
predictions = predictions.astype(int)
#predicted_val = [int(round(p)) for p in predictions]


# In[ ]:


column_names  = ['start.'+str(i) for i in range(1,401)]


# In[ ]:


submission = pd.DataFrame(data=predictions,    # values
              columns=column_names)


# In[ ]:


id_col = test[['id']]


# In[ ]:


submission['id'] = id_col


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index = False)


# That's all . However will try this example with some other way. In the meantime please provide your suggestions

# In[ ]:




