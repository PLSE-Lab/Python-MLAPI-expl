#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[ ]:


#comapring the same with kaggle data and see if we can plot the same to visualise
import os
#os.chdir("C:\\Users\\vidya\\Anaconda3")


# In[ ]:


train_df=pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_df=pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


# In[ ]:


#create a fucntion to process the kaggle images datasets
def data_preprocess(df):
    X=df.drop(columns=["label"])
    y=df["label"]
    X=np.array(X)
    X=X.reshape(-1,28,28)
    X=X/255.0
    return X,y


# In[ ]:


X_train,y_train=data_preprocess(train_df)
X_test,y_test=data_preprocess(test_df)


# In[ ]:


plt.imshow(X_train[0])
plt.show()


# In[ ]:


plt.imshow(X_test[0])
plt.show()


# In[ ]:


y_test[0]


# In[ ]:


#tensor flow takes images like image dmesnions and a channel,so our 28*28 image will be a 28*28*1 or 28*28*3 based on b/wor color image


# In[ ]:


X_train=X_train.reshape(-1,28,28,1)


# In[ ]:


X_test=X_test.reshape(-1,28,28,1)


# In[ ]:


X_train.shape


# In[ ]:


X_train.shape[1:]


# In[ ]:


#since we have multople classes we have to encode them as well
y_train=keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')


# In[ ]:


y_test=keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.activations import relu
from keras.layers import Dropout
from keras.callbacks import History
#lets use callbacks to add restrictions to model  
from keras.callbacks import EarlyStopping 
es=EarlyStopping(patience=5,restore_best_weights=True)


# In[ ]:


model=Sequential()
model.add(Conv2D(128,kernel_size=(3,3),activation="relu",input_shape=(X_train.shape[1:])))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))

model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[ ]:


history=model.fit(X_train,y_train,epochs=30,validation_split=0.2,callbacks=[es])


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


X_test=X_test.reshape(-1,28,28,1)


# In[ ]:


evaluation=model.evaluate(X_test,y_test)


# In[ ]:


evaluation


# In[ ]:


loss=evaluation[0]
accuracy=evaluation[1]
print("the loss is {} and accuracy is {}".format(loss,accuracy))


# In[ ]:


model.save("fashion_mnist_cnn170320")


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


print(np.argmax(prediction[0]))


# In[ ]:





# In[ ]:




