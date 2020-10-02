#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[18]:


train_df.info()


# In[19]:


test_df.info()


# In[20]:


sns.countplot(train_df['label'])
plt.show()


# In[21]:


# labels = pd.get_dummies(train_df['label'])
from keras.utils.np_utils import to_categorical


labels = to_categorical(train_df['label'], num_classes = 10)
train_df = train_df.drop('label',axis = 1)


# In[22]:


# labels.head(10)


# In[23]:


train_df = train_df.astype('float32')
train_df = train_df.astype('float32')
train_df/= 255
test_df/= 255


# In[24]:


train_df = train_df.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)


# In[33]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from keras.optimizers import SGD
from keras.optimizers import RMSprop

model = Sequential()

X_train, X_test, y_train, y_test = train_test_split(train_df, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(train_df,labels,batch_size=128,epochs=50,validation_split=0.1,shuffle=True,verbose=2)


# In[ ]:


res = model.predict(test_df)
res = np.argmax(res,axis = 1)

res_df = pd.DataFrame(data = {'ImageId':range(1,28001),'Label':res})

res_df.to_csv('res.csv',index=False)

