#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import tensorflow as tf
import numpy as np # math (array ; .expand_dims ; .squeeze Remove 1-dimensional entries of the shape ; )
import pandas as pd # import dataset (.read_csv ; )
import cv2
#from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt # plotting (.imshow to render images ; )
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, LSTM, CuDNNLSTM, Flatten, Reshape, ZeroPadding2D, Convolution2D, BatchNormalization, Activation  # CuDNNLSTM only on GPU
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm, tqdm_notebook
import random as rn


# In[ ]:


get_ipython().system('ls ../input/aerial-cactus-identification')


# In[ ]:


PATH = '../input/aerial-cactus-identification/'


# # Load Data

# In[ ]:


# Loading Dataframes

submissionDf = pd.read_csv(PATH + 'sample_submission.csv')
submissionDf.head()


# In[ ]:


trainDf = pd.read_csv(PATH + 'train.csv')
trainDf.head()


# In[ ]:


TRAINPATH = '../input/aerial-cactus-identification/train/train/'
TESTPATH = '../input/aerial-cactus-identification/test/test/'


# In[ ]:


# Load Images

X_tr = []
Y_tr = []

imgs = trainDf['id'].values

for img_id in tqdm_notebook(imgs):
    X_tr.append(cv2.imread(TRAINPATH + img_id))    
    Y_tr.append(trainDf[trainDf['id'] == img_id]['has_cactus'].values[0])  
    
X_tr = np.asarray(X_tr)
X_tr = X_tr.astype('float32')
X_tr /= 255
Y_tr = np.asarray(Y_tr)


# ## Data separation :
# ### * 12000 imgs for training
# ### * 3000 imgs for validation (split made in model.fit())
# ### * 2500 imgs for testing
# ### * Submission set of 4000 images for competition scoring

# In[ ]:


# Separation train set and validation set

x_tr = X_tr[:15000]
y_tr = Y_tr[:15000]

x_te = X_tr[15000:]
x_te = X_tr[15000:]


# # Build Model

# ## Model 1

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model = Sequential()

model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation= 'sigmoid'))

model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
 
history = model.fit(x_tr,y_tr,epochs=100,batch_size=20, validation_split=0.2, callbacks=[es, mc])

# history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50) 
    


# In[ ]:


print (history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  


# In[ ]:


preds = np.vectorize(lambda x: 1 if x > 0.75 else 0)(model.predict(x_te))
print(preds.shape)

preds = np.resize(preds,(2500))
print(preds.shape)


# In[ ]:


trues = trainDf.iloc[-2500:]['has_cactus'].values


# ## Score : Model 1

# In[ ]:


np.sum(preds == trues)/2500


# ## Model 2

# In[ ]:


model2 = Sequential()

model2.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu', input_shape=x_tr.shape[1:]))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Flatten())
model2.add(Dense(1, activation= 'sigmoid'))

model2.summary()


# In[ ]:


es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
mc2 = ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
 
history2 = model2.fit(x_tr,y_tr,epochs=100,batch_size=20, validation_split=0.2, callbacks=[es2, mc2])

# history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50) 


# In[ ]:


print (history2.history.keys())

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  


# In[ ]:


preds2 = np.vectorize(lambda x: 1 if x > 0.75 else 0)(model2.predict(x_te))
print(preds2.shape)

preds2 = np.resize(preds2,(2500))
print(preds2.shape)


# ## Score : Model 2

# In[ ]:


np.sum(preds2 == trues)/2500


# In[ ]:


get_ipython().system('ls')


# In[ ]:


from tensorflow.keras.models import load_model


# ## Score : Saved Model 1

# In[ ]:


saved_model = load_model('best_model.h5')

preds3 = np.vectorize(lambda x: 1 if x > 0.75 else 0)(saved_model.predict(x_te))
print(preds3.shape)

preds3 = np.resize(preds3,(2500))
print(preds3.shape)

np.sum(preds3 == trues)/2500


# ## Score : Saved Model 2

# In[ ]:


saved_model2 = load_model('best_model2.h5')

preds4 = np.vectorize(lambda x: 1 if x > 0.75 else 0)(saved_model2.predict(x_te))
print(preds4.shape)

preds4 = np.resize(preds4,(2500))
print(preds4.shape)

np.sum(preds4 == trues)/2500


# ## Model with batch normalization

# In[ ]:


model3 = Sequential()

model3.add(Conv2D(32, (3, 3), input_shape=x_tr.shape[1:]))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Conv2D(32, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Conv2D(32, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(64, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Conv2D(64, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Conv2D(64, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(128, (3, 3)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))

model3.add(Flatten())
model3.add(Dense(1024))
model3.add(Activation('relu'))
model3.add(Dropout(0.6))

model3.add(Dense(256))
model3.add(Activation('relu'))
model3.add(Dropout(0.6))

model3.add(Dense(1))
model3.add(Activation('sigmoid'))

model3.summary()


# In[ ]:


es3 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
mc3 = ModelCheckpoint('best_model3.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
 
history3 = model3.fit(x_tr,y_tr,epochs=100,batch_size=20, validation_split=0.2, callbacks=[es3, mc3])

# history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50) 


# In[ ]:


print (history3.history.keys())

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  


# ## Score : Model 3

# In[ ]:


preds3 = np.vectorize(lambda x: 1 if x > 0.75 else 0)(model3.predict(x_te))
print(preds3.shape)

preds3 = np.resize(preds3,(2500))
print(preds3.shape)

np.sum(preds3 == trues)/2500


# In[ ]:


get_ipython().system('ls -l')


# In[ ]:


saved_model3 = load_model('best_model3.h5')

preds33 = np.vectorize(lambda x: 1 if x > 0.75 else 0)(saved_model3.predict(x_te))
print(preds33.shape)

preds33 = np.resize(preds33,(2500))
print(preds33.shape)

np.sum(preds33 == trues)/2500


# In[ ]:


bestModel = saved_model3


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_tst = []\nTest_imgs = []\n\nimgs = submissionDf['id'].values\n\nfor img_id in tqdm_notebook(imgs):\n    X_tst.append(cv2.imread(TESTPATH + img_id))     \n    Test_imgs.append(img_id)\n    \nX_tst = np.asarray(X_tst)\nX_tst = X_tst.astype('float32')\nX_tst /= 255\n")


# In[ ]:


test_predictions = bestModel.predict(X_tst)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])
sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)


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


# In[ ]:




