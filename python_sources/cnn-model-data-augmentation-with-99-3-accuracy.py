#!/usr/bin/env python
# coding: utf-8

# Import Pandas Library

# In[ ]:


import pandas as pd


# Read training & test dataset

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Get label dataset from training set

# In[ ]:


label = train_df["label"]


# Drop label set from training set

# In[ ]:


train_df = train_df.drop(["label"], axis=1)


# Normalise dataset

# In[ ]:


train_df = train_df.astype('float32') / 255
test_df = test_df.astype('float32') / 255


# Reframe datset to fit into (28,28,1) structure

# In[ ]:


train_images = train_df.values.reshape((42000,28,28,1))
test_images = test_df.values.reshape((28000,28,28,1))


# Import model_selection from sklearn & solit training set into training & validation set

# In[ ]:


from sklearn import model_selection


# In[ ]:


train_x,valid_x,train_y,valid_y = model_selection.train_test_split(train_images,label,
                                                                   test_size=0.1,stratify=label,random_state=0)


# Import to_categorical from Keras for One hot encoding of label set

# In[ ]:


from keras.utils import to_categorical


# In[ ]:


train_labels = to_categorical(train_y)
test_labels = to_categorical(valid_y)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)


# Import Keras Library

# In[ ]:


from keras import models
from keras import layers
from keras.layers.normalization import BatchNormalization


# Built a CNN model with droput & batch normalization

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dense(10,activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(gen.flow(train_x,train_labels, batch_size=64),
                              epochs = 20, validation_data = (valid_x,test_labels),
                              steps_per_epoch=train_x.shape[0] // 64)


# In[ ]:


#model.fit(train_x,train_labels,epochs=20,batch_size=64,validation_data=(valid_x, test_labels))


# In[ ]:


valid_loss, valid_acc = model.evaluate(valid_x, test_labels)
valid_acc


# In[ ]:


pred_test = model.predict(test_images)
ytestpred = pred_test.argmax(axis=1)


# In[ ]:


df = pd.read_csv('../input/sample_submission.csv')
df['Label'] = ytestpred
df.head()


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:




