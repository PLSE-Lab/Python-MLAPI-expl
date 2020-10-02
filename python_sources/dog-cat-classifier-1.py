#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tqdm import tqdm
import zipfile
import cv2
import random
from sklearn.model_selection import train_test_split 
TrainFile='../working/train/train/'
TestFile='/dev/shm/test'
IMG_SIZE=128


# In[ ]:


numImages=25000
nH = IMG_SIZE
nW = IMG_SIZE
nC = 3


# In[ ]:



unzp = get_ipython().getoutput('unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train')


# In[ ]:


def findLabel(title):
    if title=='dog':
        return 0
    else :
        return 1


# In[ ]:


def trainPrep(TrainFile): 
    train_data=[]
    for file in tqdm(os.listdir(TrainFile)):
        label,name,_=file.split('.')

        y=findLabel(label)
        img= os.path.join(TrainFile,file)
        x=cv2.imread(img)

        imgArr = cv2.resize(x, (IMG_SIZE,IMG_SIZE))
        
        train_data.append([imgArr,y])

    random.shuffle(train_data)
    
    #np.save('train_data.npy',train_data)
    return np.array(train_data)



    
    


# In[ ]:


train_data=trainPrep(TrainFile)


# In[ ]:


#os.listdir('../working/train/train/')


# In[ ]:


train_data=train_data.T


# In[ ]:


from matplotlib.pyplot import imshow

imshow(train_data[0][600])


# In[ ]:


y=train_data[1]
y=y.reshape(numImages,1)
y.shape


# In[ ]:


X= np.array([i for i in train_data[0]]).reshape(-1,nH,nW,nC)
X.shape


# In[ ]:



from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dropout,MaxPooling2D
from keras.optimizers import adam
from keras.initializers import random_normal
import keras


# In[ ]:


stride=2
mpstride=2
kernelM=(3,3)
kernelC=(4,4)
nF=16
for i in [1]:
    print('for stride: ', i)
    
    model = Sequential()
    
    model.add(Conv2D(nF,kernel_size=kernelC,strides = stride,activation='relu',padding='same', input_shape=(nH,nW,nC)))
    model.add(MaxPooling2D(pool_size=kernelM,strides=mpstride,padding='same'))
    
    model.add(Conv2D(nF*2,kernel_size=3,strides = stride,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=mpstride,padding='same'))
    
    
    model.add(Flatten())
    
    model.add(Dense(500,kernel_initializer=random_normal(),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200,kernel_initializer=random_normal(),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(70 ,kernel_initializer=random_normal(),activation='relu'))
    #model.add(Dense(55,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=adam(learning_rate=0.0001),metrics=['accuracy'])

    model.fit(X,y,batch_size=16,epochs=20,validation_split = 0.07)
    


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


datagen = FixedImageDataGenerator(horizontal_flip = True,
                                 
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                                                         
                            )
valdatagen = FixedImageDataGenerator()
it = datagen.flow(X_train,y_train)
itval= valdatagen.flow(X_val,y_val)
'''steps_per_epoch=25000//16,'''
model.fit_generator(it,epochs=20,steps_per_epoch=50,validation_data=itval)

