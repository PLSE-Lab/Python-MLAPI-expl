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


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import pandas as pd 
df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
df.head()


# In[ ]:


sick= df[df.Target==1]
sick= sick[['patientId','Target']]
sick.head()


# In[ ]:


sickfiles=[]
for idx , row  in sick.iterrows():
    name = row['patientId']
    sickfiles.append(name)


# In[ ]:


print(len(sickfiles))


# In[ ]:


get_ipython().system('pip install pydicom')


# In[ ]:


from keras.utils import Sequence
import pydicom, numpy as np
from skimage.transform import resize
class generator(Sequence):
    
    def __init__(self, folder, filenames,sickfiles,batch_size=32, image_size=256, shuffle=True):
        self.folder = folder
        self.filenames = filenames
        self.sickfiles=sickfiles
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.sickfiles:
            y=1
        else :
            y=0    
        img = resize(img, (256, 256), anti_aliasing=True)
        img=img*(1./255)
        img = np.expand_dims(img, -1)
        return img, y
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        items = [self.__load__(filename) for filename in filenames]
        imgs, y = zip(*items)
        imgs = np.array(imgs)
        y = np.array(y)
        return imgs, y
         
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filenames)
        
    def __len__(self):
        return int(len(self.filenames) / self.batch_size)


# In[ ]:


folder = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
filenames = os.listdir(folder)
np.random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 5000
n_train_samples = len(filenames) - n_valid_samples
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))


# In[ ]:


train_gen = generator(folder, train_filenames,
                      sickfiles, batch_size=32,
                      image_size=256, shuffle=True)
valid_gen = generator(folder, valid_filenames, 
                      sickfiles, batch_size=32, 
                      image_size=256, shuffle=False)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)
model.fit_generator(train_gen,validation_data=valid_gen,epochs=10,callbacks=[checkpoint])

