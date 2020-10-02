#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install efficientnet')


# In[ ]:


import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


PATH = '../input/losers' 
train_df = pd.read_csv('../input/losers/train.csv',dtype={'category':str})
test_df = pd.read_csv('../input/losers/test.csv')

train_df['filename'] = '../input/losers/train/train/'+train_df['category']+'/'+ train_df['filename']
test_df['filename'] = '../input/losers/test/test/'+ test_df['filename']


# In[ ]:


img = cv2.imread(train_df['filename'][1])
plt.imshow(img)


# ## Image Data Generator

# In[ ]:


train, val = train_test_split(train_df, test_size = 0.2)


# In[ ]:


len(train_df),len(train),len(val)


# In[ ]:


train.head()


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator

train_data_gen= ImageDataGenerator(rescale=1/255)


# In[ ]:


IMG_SIZE=300
BATCH_SIZE=16


# In[ ]:


train_generator=train_data_gen.flow_from_dataframe(train,directory=None,
                                                      target_size=(IMG_SIZE,IMG_SIZE),
                                                      x_col="filename",
                                                      y_col='category',
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                        subset='training',
                                                      batch_size=BATCH_SIZE)


# In[ ]:


val_generator=train_data_gen.flow_from_dataframe(val,directory=None,
                                                      target_size=(IMG_SIZE,IMG_SIZE),
                                                      x_col="filename",
                                                      y_col='category',
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE,
                                                  )


# In[ ]:


x,y = train_generator.next()
for i in range(0,3):
    image = x[i]
    label = y[i]
    print (label)
    plt.imshow(image)
    plt.show()


# ## Training Model

# In[ ]:


from keras.models import Sequential,Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization,Input,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
import efficientnet.keras as efn 
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import keras


# In[ ]:



model =efn.EfficientNetB3(weights = 'imagenet', include_top=False, input_shape = (IMG_SIZE,IMG_SIZE,3))


# In[ ]:


x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(42, activation="softmax")(x)

model = Model(inputs=model.input, outputs=predictions)


# In[ ]:


model.summary()


# In[ ]:



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


results = model.fit_generator(train_generator,epochs=10,
                              steps_per_epoch=train_generator.n/BATCH_SIZE,
                              validation_data=val_generator,
                             validation_steps=val_generator.n/BATCH_SIZE,
                              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=3, min_lr=0.000001)])


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

