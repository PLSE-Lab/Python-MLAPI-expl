#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-addons')


# In[ ]:


get_ipython().system('pip install efficientnet')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train_df=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
train_df['image_id']=train_df['image_id']+'.jpg'
#train_df['label']=np.argmax(train_df.iloc[:,1::].values,axis=1).astype('str')
train_df.head()


# In[ ]:


test_df=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
test_df['image_id']=test_df['image_id']+'.jpg'
# test_df['healthy']=np.zeros(len(test_df)).astype(int)
# test_df['multiple_diseases']=np.zeros(len(test_df)).astype(int)
# test_df['rust']=np.zeros(len(test_df)).astype(int)
# test_df['scab']=np.zeros(len(test_df)).astype(int)

test_df.head()


# ### Image Data Generator

# In[ ]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(train_df, test_size = 0.2)


# In[ ]:


len(train_df),len(train),len(val)


# In[ ]:


train.head()


# In[ ]:


val.head()


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator

train_data_gen= ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    rescale=1/255,
    fill_mode='nearest',
    shear_range=0.1,
    brightness_range=[0.5, 1.5])


# In[ ]:


img_shape=300
batch_size=16


# In[ ]:


train_generator=train_data_gen.flow_from_dataframe(train,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(img_shape,img_shape),
                                                      x_col="image_id",
                                                      y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',
                                                      shuffle=False,
                                                       subset='training',
                                                      batch_size=batch_size)


# In[ ]:





# In[ ]:


val_generator=train_data_gen.flow_from_dataframe(val,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(img_shape,img_shape),
                                                      x_col="image_id",
                                                      y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',
                                                      shuffle=False,
                                                      batch_size=batch_size,
                                                  )


# In[ ]:


test_generator=train_data_gen.flow_from_dataframe(test_df,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(img_shape,img_shape),
                                                      x_col="image_id",
                                                      y_col=None,
                                                      class_mode=None,
                                                      shuffle=False,
                                                      batch_size=batch_size)


# In[ ]:


train_generator.next()[0].shape,train_generator.next()[1].shape


# In[ ]:


# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization,Input,MaxPooling2D,GlobalMaxPooling2D,concatenate
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow as tf


# In[ ]:


from keras.models import Sequential,Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization,Input,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
import efficientnet.keras as efn 
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import keras


# In[ ]:



model =efn.EfficientNetB4(weights = 'imagenet', include_top=False, input_shape = (img_shape,img_shape,3))


# In[ ]:


x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(4, activation="softmax")(x)


# In[ ]:


model = Model(inputs=model.input, outputs=predictions)


# In[ ]:


def custom_loss(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)


# In[ ]:


import tensorflow_addons as tfa


# In[ ]:


model.compile(optimizer='adam', loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=['accuracy'])


# In[ ]:


y_true=val.iloc[:,1::].values


# In[ ]:


results = model.fit_generator(train_generator,epochs=15,
                              steps_per_epoch=train_generator.n/batch_size,
                              validation_data=val_generator,
                             validation_steps=val_generator.n/batch_size,
                              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)])


# In[ ]:


model.save_weights('model_weights.h5')


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#model.load_weights('model.h5')


# In[ ]:


y_pred = model.predict_generator(val_generator,steps=val_generator.n/batch_size)
y_pred=y_pred.round().astype(int)


# In[ ]:


print(accuracy_score(y_true,y_pred))
print(f1_score(y_true,y_pred,average='macro'))
print(roc_auc_score(y_true,y_pred,average='macro'))


# In[ ]:


y_test=model.predict(test_generator,steps=test_generator.n/batch_size)


# In[ ]:


y_test.shape


# In[ ]:


sub_df=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub_df.head()


# In[ ]:


len(sub_df),len(y_test)


# In[ ]:


for i,j in enumerate(['healthy','multiple_diseases','rust','scab']):
    sub_df[j]=y_test[:,i]


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv', index=False)

