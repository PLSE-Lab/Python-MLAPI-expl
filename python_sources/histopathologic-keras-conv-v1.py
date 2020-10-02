#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import pandas as pd
import os
import numpy as np


# In[ ]:


# dimensions of our images.
img_width, img_height = 96, 96
validation_split = 0.1

train_data_dir = '../input/train'
test_data_dir = '../input/test'

train_df = pd.read_csv('../input/train_labels.csv')
train_df['filename'] = train_df['id'] + ".tif"
train_df['class'] = train_df['label']

test_df = pd.DataFrame({'filename':os.listdir(test_data_dir)})

nb_train_samples = train_df.shape[0] - train_df.shape[0]*validation_split
nb_validation_samples = nb_train_samples*validation_split
nb_test_samples = test_df.shape[0]

epochs = 2
batch_size = 35

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[ ]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    #preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    featurewise_center=True, 
    featurewise_std_normalization=True,
    zca_whitening=True,
    vertical_flip=True,
    validation_split=validation_split)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_data_dir,
    target_size = (img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    subset="training",
    class_mode = 'binary')

valid_generator=train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_data_dir,
    target_size = (img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    subset="validation",
    class_mode = 'binary')

test_generator = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_data_dir,
    target_size = (img_width, img_height),
    shuffle=False,
    batch_size=batch_size,
    class_mode = None)


# In[ ]:


kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.5

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (img_width, img_height, 3)))
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation = "sigmoid"))

# Compile the model
model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])


# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n//valid_generator.batch_size,
    callbacks=[reducel, earlystopper])


# In[ ]:


model.evaluate_generator(generator=valid_generator,  steps= nb_validation_samples / batch_size, verbose=1)


# In[ ]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")


# In[ ]:


test_generator.reset()

pred=model.predict_generator(test_generator, steps= nb_test_samples / batch_size, verbose=1)
print('=== raw preds ===')
print(pred[0:10])


# In[ ]:



pred_list = [int(round(pred[i][0])) for i in range(0, pred.shape[0])]
pred_list

filenames=test_generator.filenames
filenames = [f.split(sep='.')[0] for f in filenames]
results=pd.DataFrame({"id":filenames,
                      "label":pred_list})
results.to_csv("results.csv",index=False)

