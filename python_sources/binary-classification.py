#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


img_width, img_height = 150, 150

train_data_dir = '../input/binary-data/binary_data/train'
validation_data_dir = '../input/binary-data/binary_data/validation'
nb_train_samples = 9920
nb_validation_samples = 2483
epochs = 25
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
    #color_mode="grayscale")


# In[ ]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
    #color_mode="grayscale")


# In[ ]:


get_ipython().run_cell_magic('time', '', "h = model.fit_generator(\n    train_generator,\n    steps_per_epoch=nb_train_samples // batch_size,\n    epochs=epochs,\n    validation_data=validation_generator,\n    validation_steps=nb_validation_samples // batch_size,\n    callbacks=[\n        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),\n        tf.keras.callbacks.ModelCheckpoint(filepath = '/kaggle/working/model_{val_accuracy:.3f}.h5', save_best_only=True,\n                                          save_weights_only=False, monitor='val_accuracy')\n    ])\n\n#model.save_weights('first_try.h5')")


# In[ ]:


accs = h.history['loss']
val_accs = h.history['val_loss']

plt.plot(range(len(accs)),accs, label = 'Training')
plt.plot(range(len(accs)),val_accs, label = 'Validation')
plt.legend()
plt.show()


# In[ ]:


accs = h.history['accuracy']
val_accs = h.history['val_accuracy']

plt.plot(range(len(accs)),accs, label = 'Training')
plt.plot(range(len(accs)),val_accs, label = 'Validation')
plt.legend()
plt.show()


# In[ ]:


test_model = load_model('./model_0.931.h5')
img = load_img('../input/testset2/s.jpeg', grayscale=False, target_size=(img_width,img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
#prob = test_model.predict_proba(x)
print(preds)


# In[ ]:




