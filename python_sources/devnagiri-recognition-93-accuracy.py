#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# In[ ]:


data =pd.read_csv('../input/devanagari-character-set/data.csv')


# In[ ]:


pd.pandas.set_option('display.max_columns',None)


# In[ ]:


data.head()


# In[ ]:


data.groupby("character").count()


# In[ ]:


#Pixel Distribution Of The Data set
plt.hist(data.iloc[0,:-1])
plt.show()


# In[ ]:


X=data.iloc[:,:-1]/255
y=data.iloc[:,-1].values


# In[ ]:


y


# In[ ]:


#Visualizing The Data
char_names=data.character.unique()


# In[ ]:


char_names


# In[ ]:


char_names = data.character.unique()  
rows =10;columns=5;
fig, ax = plt.subplots(rows,columns, figsize=(8,16))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        if columns*row+col < len(char_names):
            x = data[data.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)
            x = x.astype("float64")
            x/=255
            ax[row,col].imshow(x, cmap="binary")
            ax[row,col].set_title(char_names[columns*row+col].split("_")[-1])

            
plt.subplots_adjust(wspace=1, hspace=1)   
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)
y_train=to_categorical(y_train,num_classes=46)
y_test=to_categorical(y_test,num_classes=46)


# In[ ]:


y_train


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


img_height_rows = 32
img_width_cols = 32


# In[ ]:


im_shape = (img_height_rows, img_width_cols, 1)
X_train = X_train.values.reshape(X_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
X_test = X_test.values.reshape(X_test.shape[0], *im_shape)


# In[ ]:


model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(32,32,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='Same'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='Same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=46,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))


# In[ ]:


plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.legend(loc='best')
plt.show()


# In[ ]:


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.legend(loc='best')
plt.show()


# In[ ]:





# In[ ]:




