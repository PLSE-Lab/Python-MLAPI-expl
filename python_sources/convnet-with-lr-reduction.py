#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop

train_dir = pd.read_csv('../input/train.csv')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test_file = train_dir.iloc[100,0]
im = plt.imread('../input/train/train/%s'%test_file)
plt.imshow(im)
plt.show()

print(im.shape)


# In[ ]:


train_dir.describe()


# In[ ]:


y_train = train_dir.iloc[:,1].values
X_train = np.zeros((17500,32,32,3))

im_list = train_dir.iloc[:,0]

idx = 0
for fp in im_list:
    image = plt.imread('../input/train/train/%s'%fp)
    X_train[idx,:,:,:] = image
    
    idx+=1


# In[ ]:


plt.imshow(X_train[10,:,:,:]/255)
plt.show()

print(y_train[10])


# In[ ]:


X_train_scaled = X_train/255


# In[ ]:


# lets create a convnet

cactus = Sequential()

cactus.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='Same', input_shape=(32,32,3)))
cactus.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='Same'))
cactus.add(MaxPool2D(pool_size=(2,2)))
cactus.add(Dropout(0.2))

cactus.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))
cactus.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))
cactus.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
cactus.add(Dropout(0.2))

cactus.add(Flatten())
cactus.add(Dense(256, activation='relu'))
cactus.add(Dropout(0.50))

cactus.add(Dense(1, activation='sigmoid'))

opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
lrreduce = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

cactus.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


cactus.summary()


# In[ ]:


estop = EarlyStopping(patience=3)


# In[ ]:


cactus.fit(X_train_scaled, y_train,
          validation_split=0.15,
          verbose=True,
          epochs=30,
          batch_size=100,
          callbacks=[lrreduce]
)


# In[ ]:


import os
fileid = []

def read_in_test(dirstr):
    out_array = np.zeros((4000,32,32,3))
    
    dir_p = os.fsencode(dirstr)
    
    idx=0
    for file in os.listdir(dir_p):
        filename = os.fsdecode(file)
        
        fileid.append(filename)
        
        out_array[idx,:,:,:] = plt.imread('../input/test/test/%s'%filename)
        idx+=1
    
    return out_array/255


# In[ ]:


X_test = read_in_test('../input/test/test')


# In[ ]:


out = cactus.predict_proba(X_test)


# In[ ]:


out.ravel().shape


# In[ ]:


sub = pd.DataFrame({'id': fileid, 'has_cactus': out.ravel()})
sub['has_cactus'] = sub['has_cactus'].apply(lambda x: 1 if x>0.50 else 0)


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('cactus_submission.csv',index=False)


# In[ ]:




