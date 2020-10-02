#!/usr/bin/env python
# coding: utf-8

# In[19]:


from keras.layers import Dense,Dropout,Input,GlobalAveragePooling2D,MaxPooling2D,Add,concatenate
from keras.callbacks import EarlyStopping
from keras.models import Model
#from keras.applications.inception_v3 import InceptionV3 ,preprocess_input
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.losses import categorical_crossentropy,binary_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_colwidth=150
import pickle


# In[20]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir('../input/'))


# In[21]:


#Load the data.
train = pd.read_json("../input/train.json")
train.head()


# In[22]:


train.info()


# In[23]:


#train.inc_angle.value_counts()


# In[24]:


#train[train['inc_angle']=='na']


# In[25]:


train.inc_angle.replace(to_replace='na',value=0.0,inplace=True)


# In[26]:


#train.inc_angle.value_counts()


# In[27]:


train.info()


# In[28]:


test=pd.read_json('../input/test.json')
test.head()


# In[29]:


test.info()


# In[30]:


len(train.band_1[0])


# In[31]:


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X1 = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
X2 = np.array(train.inc_angle)
y = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X1_test_data = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
X2_test_data = np.array(test.inc_angle)


# In[32]:


print(X1.shape)
print(X2.shape)
print(y.shape)


# In[33]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,random_state=123, test_size=0.20)
print(X1_train.shape)
print(X2_train.shape)
print(y_train.shape)
print(X1_test.shape)
print(X2_test.shape)
print(y_test.shape)


# In[34]:


bsize=32
num_train_sample=len(X1_train)
num_test_sample=len(X1_test)
input_shape1=X1_train[0].shape
input_shape2=X2_train.shape
print(num_train_sample)
print(num_test_sample)
print(input_shape1)
print(input_shape2)


# In[17]:


# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
#                                  rescale=1./255,
#                                  rotation_range=30, width_shift_range=0.2, 
#                                  height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True
#                                 )


# In[18]:


# test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
#                                rescale=1./255)


# In[19]:


# training_set=train_datagen.flow(X1_train,y=y_train,batch_size=bsize)
# testing_set=test_datagen.flow(X1_test,y=y_test,batch_size=bsize)


# In[20]:


# base_model = VGG16(weights = 'imagenet', include_top = False, input_shape=input_shape1)


# In[21]:


# x1 = base_model.output

# x1 = GlobalAveragePooling2D()(x1)
# x1 = Dense(512, activation='relu')(x1)


# In[22]:


# input2=Input(shape=(1,))
# x2=Dense(512,activation='relu')(input2)


# In[23]:


# added = Add()([x1, x2])


# In[24]:


# x=Dense(512, activation='relu')(added)
# x=Dropout(0.2)(x)
# x=Dense(256, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)


# In[25]:


# # The model we will train
# model = Model(inputs = [base_model.input,input2], outputs = predictions)


# In[26]:


# # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False


# In[27]:


# # Compile with Adam
# model.compile(Adam(lr=.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[28]:


# my_callback=[EarlyStopping(monitor='val_acc', min_delta=0.0005, patience=5, verbose=1, mode='auto')]


# In[30]:


# history=model.fit([X1_train,X2_train],y_train,batch_size=bsize,epochs=100,verbose=1,callbacks=my_callback,
#                   validation_data=([X1_test,X2_test],y_test))


# ###  Submission file

# In[31]:


# sub_prediction = model.predict([X1_test_data, X2_test_data], verbose=1, batch_size=200)


# In[32]:


# sub_prediction.shape


# In[33]:


# submission = pd.DataFrame({'id': test["id"], 'is_iceberg': sub_prediction.reshape((sub_prediction.shape[0]))})
# submission.head(10)


# In[34]:


# submission.to_csv("sub1.csv", index=False)


# In[ ]:




