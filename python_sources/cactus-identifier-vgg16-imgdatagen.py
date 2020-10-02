#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_dir = "../input/train/train/"
test_dir = "../input/test/test/"
train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


im = cv2.imread("../input/train/train/01e30c0ba6e91343a12d2126fcafc0dd.jpg")
plt.imshow(im)


# In[ ]:


ResNet50_net = VGG16(weights='imagenet', 
                     include_top=False, 
                     input_shape=(32, 32, 3))


# In[ ]:


ResNet50_net.trainable = False
ResNet50_net.summary()


# In[ ]:


model = Sequential()
model.add(ResNet50_net)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# def sigmoid(x):
    # return 1.0/(1.0 + np.exp(-x))


# In[ ]:


# def binary_crossentropy(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_true, K.sigmoid(y_pred)), axis=-1)


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])


# In[ ]:


X_tr = []
Y_tr = []
imges = train_df['id'].values
for img_id in tqdm_notebook(imges):
    X_tr.append(cv2.imread(train_dir + img_id))    
    Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  
X_tr = np.asarray(X_tr)
X_tr = X_tr.astype('float32')
X_tr /= 255
Y_tr = np.asarray(Y_tr)


# In[ ]:


batch_size = 32
nb_epoch = 1000


# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset\n",
                             samplewise_center=False,  # set each sample mean to 0\n",
                             featurewise_std_normalization=False,  # divide inputs by dataset std\n",
                             samplewise_std_normalization=False,  # divide each input by its std\n",
                             zca_whitening=False,  # apply ZCA whitening\n",
                             zca_epsilon=1e-06,  # epsilon for ZCA whitening\n",
                             rotation_range=40,  # SET TO 0 IF NEEDED # randomly rotate images in 0 to 180 degrees\n",
                             width_shift_range=0.2,  # randomly shift images horizontally\n",
                             height_shift_range=0.2,  # randomly shift images vertically\n",
                             shear_range=0.2,  # set range for random shear\n",
                             zoom_range=0.2,  # set range for random zoom\n",
                             channel_shift_range=0.,  # set range for random channel shifts\n",
                             fill_mode='nearest',
                             validation_split=0.1)  # set mode for filling points outside the input

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_tr)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=batch_size, subset='training'), validation_data=datagen.flow(X_tr, Y_tr, batch_size=batch_size, subset='validation'), steps_per_epoch=len(X_tr) / batch_size, epochs=nb_epoch, shuffle=True)


# In[ ]:


pretrained = True

"""
%%time
# Train model
history = model.fit(X_tr, Y_tr,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)
"""


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_tst = []\nTest_imgs = []\nfor img_id in tqdm_notebook(os.listdir(test_dir)):\n    X_tst.append(cv2.imread(test_dir + img_id))     \n    Test_imgs.append(img_id)\nX_tst = np.asarray(X_tst)\nX_tst = X_tst.astype('float32')\nX_tst /= 255")


# In[ ]:


# Prediction
test_predictions = model.predict(X_tst) # sigmoid(model.predict(X_tst))


# In[ ]:


sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])


# In[ ]:


sub_df['id'] = ''
cols = sub_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub_df=sub_df[cols]


# In[ ]:


for i, img in enumerate(Test_imgs):
    sub_df.set_value(i,'id',img)


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv',index=False)

