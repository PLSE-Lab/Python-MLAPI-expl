#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow.keras.datasets import cifar10


# In[ ]:


(X_train,y_train),(X_test,y_test)=cifar10.load_data()


# In[ ]:


X_train.shape


# In[ ]:


X_train[0].shape


# In[ ]:


plt.imshow(X_train[0])


# In[ ]:


plt.imshow(X_train[12])


# In[ ]:


X_train[0].max()


# In[ ]:


X_train=X_train/255


# In[ ]:


X_test=X_test/255


# In[ ]:


X_test.shape


# In[ ]:


y_test


# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


y_cat_train=to_categorical(y_train,10)
y_cat_test=to_categorical(y_test,10)


# In[ ]:


y_train[0]


# In[ ]:


plt.imshow(X_train[0])


#  <br><br>This is the final output sheet representing values corresponding to respective images
#  
#  <font face = "Verdana" size ="1">
#     <img src='https://corochann.com/wp-content/uploads/2017/04/cifar10_plot.png'>
# 
#     

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[ ]:


28*28


# In[ ]:


32*32*3


# In[ ]:


model=Sequential()

#1st layer
#Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation="relu"))

#Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

#2nd Layer
#Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation="relu"))

#Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop=EarlyStopping(monitor="val_loss",patience=2)


# In[ ]:


model.fit(X_train,y_cat_train,epochs=15,validation_data=(X_test,y_cat_test),callbacks=[early_stop])


# In[ ]:


metrics=pd.DataFrame(model.history.history)


# In[ ]:


metrics.columns


# In[ ]:


metrics[["accuracy","val_accuracy"]].plot()


# In[ ]:


metrics[["loss","val_loss"]].plot()


# In[ ]:


model.evaluate(X_test,y_cat_test,verbose=0)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


predictions=model.predict_classes(X_test)


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


import seaborn as sns
   
plt.figure(figsize=(20,20))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)


# In[ ]:


my_image=X_test[0]


# In[ ]:


plt.imshow(my_image)


# In[ ]:


y_test[0]


# In[ ]:


model.predict_classes(my_image.reshape(1,32,32,3))


# In[ ]:


#As we can see its working well,so we are done here for now!


# In[ ]:




