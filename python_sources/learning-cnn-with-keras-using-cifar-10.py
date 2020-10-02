#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
import keras


from keras.models import Sequential 
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[ ]:



(X_train, y_train), (X_test, y_test) = cifar10.load_data() 


# In[ ]:



X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_train[1]


# In[ ]:


X_train[0]
plt.imshow(X_train[1])


# In[ ]:


y_test.shape


# In[ ]:


for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(X_train[i])
plt.show()


# In[ ]:


for i in range(0,15):
    #plt.subplot(330+2+i)
    plt.imshow(X_train[i])
    plt.show()


# In[ ]:


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
number_cat=10


# In[ ]:


y_train


# In[ ]:


y_train=keras.utils.to_categorical(y_train,number_cat)


# In[ ]:


y_train


# In[ ]:


y_test=keras.utils.to_categorical(y_test,number_cat)


# In[ ]:


y_test


# In[ ]:


X_train=X_train/255
X_test=X_test/255


# In[ ]:


#X_train


# In[ ]:


X_train.shape


# In[ ]:


Input_shape = X_train.shape[1:]


# In[ ]:


Input_shape


# In[ ]:


model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=Input_shape))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=10,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adagrad(lr=0.021),metrics=['accuracy'])


# In[ ]:


history=model.fit(X_train,y_train,validation_split=0.2 ,batch_size=32,epochs=100,shuffle=True)


# In[ ]:


evaluation=model.evaluate(X_test,y_test)
print('Test Accuracy: {}'.format(evaluation[1]))


# In[ ]:


predicted_classes=model.predict_classes(X_test)
predicted_classes


# In[ ]:


y_test=y_test.argmax(1)


# In[ ]:


y_test


# In[ ]:


L=8
W=8
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction= {}\nTrue={}'.format(predicted_classes[i],y_test[i]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# In[ ]:


def plotmodelhistory(modelhistory, acc='accuracy', valacc='val_accuracy'): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 
    axs[0].plot(range(1,len(modelhistory.history[acc])+1),modelhistory.history[acc]) 
    axs[0].plot(range(1,len(modelhistory.history[valacc])+1),modelhistory.history[valacc]) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(modelhistory.history[acc])+1),len(modelhistory.history[acc])/10) 
    axs[0].legend(['train', 'val'], loc='best') 
    axs[1].plot(range(1,len(modelhistory.history['loss'])+1),modelhistory.history['loss']) 
    axs[1].plot(range(1,len(modelhistory.history['val_loss'])+1),modelhistory.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(modelhistory.history['loss'])+1),len(modelhistory.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

plotmodelhistory(history)


# In[ ]:




