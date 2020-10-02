#!/usr/bin/env python
# coding: utf-8

# In[48]:


import cv2
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm, tqdm_notebook
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, DenseNet201
import matplotlib.pyplot as plt


# In[49]:


os.listdir('../input')


# In[50]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[52]:


batch_size=32
img_size = 32
nb_epochs = 100


# In[51]:


train_df['has_cactus'] = train_df['has_cactus'].astype(str)


# In[53]:


get_ipython().run_cell_magic('time', '', "train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)\ntrain_generator = train_datagen.flow_from_dataframe(\n        dataframe = train_df,        \n        directory = '../input/train/train',\n        x_col = 'id', y_col = 'has_cactus',\n        target_size=(img_size,img_size),\n        batch_size=batch_size,\n        class_mode='categorical',\n        subset='training')")


# In[54]:


validation_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,        
        directory = '../input/train/train',
        x_col = 'id', y_col = 'has_cactus',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')


# In[55]:


dense_net201 = DenseNet201(include_top=False, 
                  input_shape=(img_size, img_size, 3))


# In[56]:


model = Sequential()
model.add(dense_net201)
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()


# In[57]:


get_ipython().run_cell_magic('time', '', '# Train model\nfrom keras.callbacks import ModelCheckpoint\nmcp = ModelCheckpoint(filepath=\'model_check_path.hdf5\',monitor="val_acc", save_best_only=True, save_weights_only=False)\nhistory = model.fit_generator(\n            train_generator,\n#             steps_per_epoch = train_generator.samples // batch_size,\n            steps_per_epoch = 100,\n            validation_data = validation_generator, \n#             validation_steps = validation_generator.samples // batch_size,\n            validation_steps = 50,\n            epochs = 100,\n            verbose=2,\n            callbacks=[mcp])')


# In[58]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[59]:


from glob import glob
import cv2
imagePatches = glob('../input/test/test/*.jpg', recursive=True)
imagePatches[0].split('/')[4].split('.')[0]


# In[60]:


from glob import glob
import cv2
imagePatches = glob('../input/test/test/*.jpg', recursive=True)
x=[]
file_id = []
for img in imagePatches:
   
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (32, 32), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    file_id.append(img.split('/')[4])


# In[61]:


print(len(x))
print(len(file_id))


# In[62]:


x = np.array(x)
model.load_weights('model_check_path.hdf5')
y = model.predict(x)


# In[63]:


y=np.argmax(y,axis=1)


# In[64]:


y


# In[65]:


sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub.head()


# In[66]:


sub = pd.DataFrame()
sub['id']=pd.Series(file_id)
sub['has_cactus']=pd.Series(y) 
sub.head()


# In[67]:


sub.to_csv('sub.csv',index=False)

