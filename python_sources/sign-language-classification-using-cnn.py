#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_df=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test_df=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')


# In[ ]:


train_df


# In[ ]:


train_df.info()


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(data=train_df,x='label')


# ## Prepared a Dictionary to store the letters so that we can understand the alphabet of corresponding sign
# It is also to make a note that the cases J=9 and Z=25 are not to be considered.
# The key and corresponding value pairs are taken based on the count plot shown above.

# In[ ]:


num2alpha={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}


# In[ ]:


fig=plt.figure(figsize=(10,10))
for i in range(1,10):
    x=np.random.randint(1000)
    fig.add_subplot(3,3,i)
    plt.title('Label: {}, Letter: {}'.format(train_df.iloc[x,0],num2alpha[train_df.iloc[x,0]]))
    plt.imshow(train_df.iloc[x,1:].values.reshape(28,28),cmap='gray')
    plt.show()


# In[ ]:


x=train_df.iloc[:,1:].values
y=train_df.iloc[:,0]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler()
norm.fit(x)
transnorm=norm.transform(x)


# In[ ]:


x.shape,y.shape


# In[ ]:


from keras.utils import to_categorical
x=x.reshape(-1,28,28,1)
y=to_categorical(y,num_classes=25)


# In[ ]:


x.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2,random_state=4)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator as ImgDataGen
augmentData=ImgDataGen(rescale=1./255,
                        rotation_range=20,
                         height_shift_range=0.2,
                         width_shift_range=0.2,
                         horizontal_flip=False,
                         zoom_range=0.10)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dropout,BatchNormalization,MaxPooling2D,Dense,Flatten,Conv2D


# In[ ]:


model=Sequential()
model.add(Conv2D(128,(5,5),input_shape=(28,28,1),activation='relu',name='conv1'))
model.add(BatchNormalization())
model.add(Conv2D(128,(5,5),activation='relu',name='conv2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),name='max1'))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu',name='conv3'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu',name='conv4'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),name='max2'))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),activation='relu',name='conv5'))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2,2),name='max3'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(25,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.utils import plot_model
plot_model(model,to_file='model.png',show_shapes=True)


# In[ ]:


model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
earlystop=EarlyStopping(monitor="val_accuracy",min_delta=0,patience=10,mode='max',restore_best_weights=True)


# In[ ]:


history=model.fit_generator(augmentData.flow(x_train, y_train, batch_size = 64),steps_per_epoch = (len(x_train)*0.8 // 64),
 validation_data = augmentData.flow(x_valid, y_valid), validation_steps=(len(x_valid)*0.2)//64,epochs = 100,callbacks=[earlystop])


# In[ ]:


loss, acc = model.evaluate(augmentData.flow(x_valid, y_valid, batch_size=64, seed=2))
print("Loss: {}\nAccuracy: {}".format(loss, acc))


# In[ ]:




