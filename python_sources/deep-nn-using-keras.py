#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Lambda,Flatten
from keras.optimizers import Adam , RMSprop
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train=pd.read_csv("../input/train.csv")
print(train.shape)
train.head()


# In[3]:


test=pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# In[5]:


x_train=(train.ix[:,1:].values).astype('float32')
y_train=train.ix[:,0].values.astype('int32')

x_test=test.values.astype('float32')


# In[6]:


x_train.shape


# In[7]:


y_train.shape


# **Previewing image**

# In[ ]:


x_train=x_train.reshape(x_train.shape[0],28,28)


# In[ ]:


x_train.shape


# In[ ]:


#change index to view other images
index=678
plt.imshow(x_train[index])
print('Number is',y_train[index])


# **Better to change the channe to gray as there is no need for these color pixels.
# We will add another dimesion 1 for gray chanee.
# P.S - 3 is for RGB channel**

# In[ ]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


# In[ ]:


x_train.shape,x_test.shape


# **Preprocessing**

# In[ ]:


#One Hot Encoding 
#I guess everybody knows this otherwise google

from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train,num_classes=10)


# In[ ]:


y_train.shape


# In[ ]:


#same as above to verify as if it correct
print(y_train[index])
plt.plot(y_train[index])
plt.xticks(range(10))
plt.show()


# In[ ]:


mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px


# In[ ]:


x_train.reshape


# In[ ]:





# In[ ]:





# **NN**

# In[ ]:


np.random.seed(34)


# In[ ]:


model=Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))


# In[ ]:


#most useful function for a newbie **sobs**
model.summary()


# In[ ]:


model.compile(optimizer=RMSprop(lr=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[ ]:


from keras.preprocessing import image
gen=image.ImageDataGenerator()


# In[ ]:


X_train,X_val,Y_train,Y_val=train_test_split(x_train,y_train,test_size=0.10,random_state=34)
batches=gen.flow(X_train,Y_train,batch_size=64)
val_batches=gen.flow(X_val,Y_val,batch_size=64)


# In[ ]:


cache=model.fit_generator(batches,batches.n,nb_epoch=1,validation_data=val_batches,nb_val_samples=val_batches.n)


# In[ ]:


cache.history


# In[ ]:


model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, Y_train, batch_size=64)
history=model.fit_generator(batches, batches.n, nb_epoch=1)


# In[ ]:


history.history


# In[ ]:


preds=model.predict_classes(x_test,verbose=0)


# In[ ]:


preds[0:5]


# In[ ]:


subs=pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds})
subs.to_csv("sub1.csv",index=False,header=True)


# In[ ]:




