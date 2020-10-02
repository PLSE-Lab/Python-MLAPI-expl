#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <h1>Data Preprocessing </h1>
# reshape training And Test data to fit Conv2D shape

# In[ ]:


train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X_train = np.array(train.iloc[:,1:])
y_train = np.array(train.iloc[:,0])

X_test = np.array(test)

X_train=X_train.reshape(42000,28,28,1)

X_test=X_test.reshape(28000,28,28,1)


# <h1> Encode Labels </h1>
#     (e.g 1 to 1000000000 )

# In[ ]:


import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dense,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import cv2



#Normalization

X_train=X_train/255
X_test=X_test/255

#encoding 
encoder=OneHotEncoder()
y_train=y_train.reshape(-1,1)
y_train=encoder.fit_transform(y_train)


# <h1>LeNet-5 Architecture</h1>

# ![](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg)

# ![](https://engmrk.com/wp-content/uploads/2018/09/LeNEt_Summary_Table.jpg)

# **<h1>Build LeNet CNN</h1>**

# In[ ]:



model=Sequential()
model.add(Conv2D(6,5,5,input_shape=(28,28,1),activation="tanh"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())


model.add(Conv2D(16,5,5,activation="tanh"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())




model.add(Flatten())
model.add(Dense(120,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(84,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[ ]:



model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


# <h1> Callbacks </h1>
# > Choose the best weights and stop

# In[ ]:


from keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint=ModelCheckpoint("mnist.h5",monitor="loss",mode="min",save_best_only=True,verbose=1)
earlystop=EarlyStopping(monitor="loss",min_delta=0,patience=10,verbose=1,restore_best_weights=True)

callbacks=[earlystop,checkpoint]


# <h1> Train our model </h1>
# >  history used to draw learning curve
# 

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
with tf.device("/gpu:0"):
    history=model.fit(X_train,y_train,epochs=200,callbacks=callbacks)

#give 98.6% in epoch 10 you can try more epochs 


# > **Plot Learning curve**

# In[ ]:


plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot([x for x in range(0,len(history.history["loss"]))],history.history["loss"],color="blue")
plt.show


# > **Visualize training **

# In[ ]:


plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.plot([x for x in range(0,len(history.history["accuracy"]))],history.history["accuracy"],color="green")
plt.show


# **<h1>Test some images </h1>**

# In[ ]:


X_test=X_test.reshape(28000,28,28,1)
pred=model.predict(X_test)

X_test=X_test.reshape(28000,28,28)

import random
fig = plt.figure()



counter=1
for i in range(0,10):
    counter+=1
    rand=random.randint(0,len(X_test))
    if(i%2==0):
        counter=1
        fig = plt.figure()
    fig.add_subplot(1,2,counter).set_title(np.argmax(pred[rand]))
    plt.imshow(X_test[rand])


# In[ ]:





# In[ ]:





# In[ ]:




