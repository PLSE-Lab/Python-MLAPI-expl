#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns 
import sklearn


# In[ ]:


from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np 
from glob import glob
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, SGD, RMSprop,Adadelta


# In[ ]:


train = pd.read_csv('../input/traininglabels.csv')


# In[ ]:


X=train.image_id
y=train.has_oilpalm

train_images=[]
train_labels=[]
test_images=[]
test_labels=[]
split=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
# #Train
for train_index,test_index in split.split(X,y):
    
    for i in train_index:
        
        image=Image.open('../input/train_images/train_images/'+X.loc[i]).resize((100,100))
        image=np.asarray(image)/255
        train_images.append(image)
        train_labels.append(y.loc[i])

    


# In[ ]:


for i in test_index:
        image=Image.open('../input/train_images/train_images/'+X.loc[i]).resize((100,100))
        image=np.asarray(image)/255
        test_images.append(image)
        test_labels.append(y.loc[i])


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(train_images[0])


# In[ ]:


#plt.imshow(test_images[2])


# In[ ]:


train_images=np.array(train_images)
test_images=np.array(test_images)


# In[ ]:


#train_images=train_images.reshape((-1,100,100,1))
#test_images=test_images.reshape((-1,100,100,1))


# In[ ]:


train_images.shape


# In[ ]:


test_images.shape


# In[ ]:


test_labels=keras.utils.to_categorical(test_labels)
train_labels=keras.utils.to_categorical(train_labels)


# In[ ]:


from keras.applications import VGG16


# In[ ]:


conv_base = VGG16(weights = 'imagenet',
                     include_top = False,
                     input_shape= (100,100,3))


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras import layers,models


# In[ ]:


model = models.Sequential()


# In[ ]:


model.add(conv_base)
model.add(layers.Flatten())


# In[ ]:


model.add(Dense(400,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(500,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dense (2,activation='softmax'))

#from keras.callbacks import ModelCheckpoint

# checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
# model.fit(train_images,train_labels,verbose=1,
#          batch_size=train_images.shape[0]//128,
#          epochs=20,
#          validation_data=(test_images,test_labels),
#          callbacks=[checkpointer])


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])


# In[ ]:


history = model.fit(train_images,train_labels,verbose=1,
          batch_size=train_images.shape[0]//128,
          epochs=10,
          validation_data=(test_images,test_labels),
          )


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


from glob import glob
test_data=[]
test_id=[]
for i in glob('../input/leaderboard_test_data/leaderboard_test_data/*'):
    image=Image.open(i).resize((100,100))
    image=np.asarray(image)/255
    test_data.append(image)
    test_id.append(i)
    


# In[ ]:


holdout_data=[]
holdout_image_id=[]
for i in glob('../input/leaderboard_holdout_data/leaderboard_holdout_data/*'):
    image=Image.open(i).resize((100,100))
    image=np.asarray(image)/255
    holdout_data.append(image)
    holdout_image_id.append(i)


# In[ ]:


test_data=np.array(test_data)
#test_data=test_data.reshape((-1,220,220,1))
holdout_data=np.array(holdout_data)
#holdout_data=holdout_data.reshape((-1,220,220,1))


# In[ ]:


test_data.shape


# In[ ]:


prediction1=model.predict(test_data)
prediction2=model.predict(holdout_data)
predictions=np.concatenate((prediction1,prediction2))


# In[ ]:


predictions=np.argmax(predictions,axis=-1)


# In[ ]:





# In[ ]:


image_id=[i[-17:] for i in test_id]+[i[-17:] for i in holdout_image_id]


# In[ ]:


predictions


# In[ ]:


submission=pd.DataFrame({'image_id':image_id,'has_oilpalm':predictions})


# In[ ]:





# In[ ]:


submission.to_csv('pretrained_models.csv',index=False)


# In[ ]:





# In[ ]:




