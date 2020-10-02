#!/usr/bin/env python
# coding: utf-8

# ### Kaggle Invasive Species Monitoring: Get 0.97 accuracy with minimal effort.
# Finetune VGG16 top layers with Keras as described by Francois Chollet here:
# 
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# *chmaxx _ 26.5.17*

# In[ ]:


get_ipython().run_line_magic('reset', '')
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from keras.callbacks import TensorBoard
from keras_tqdm import TQDMNotebookCallback
from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

from datetime import datetime
import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
pd.options.display.max_rows = 40


# Import Labels and check distribution.

# In[ ]:


file = "D:/KI/01_keras/_kaggle/_invasiveplants/train_labels.csv"
df = pd.read_csv(file, sep=",", error_bad_lines= True)
df.invasive.value_counts()


# Separate Images according to their labels. Move them to either class folder **false/** or **true/**.

# After separating the images into two classes I manually moved 400 images to the validation folder (again with separate folders for the two classes).

# In[ ]:


get_ipython().run_line_magic('cd', '_kaggle/_invasiveplants/_train')
names = df.name
labels = df.invasive

for idx, label in enumerate(labels):
    iname = str(names[idx]) + ".jpg"
    if (label == 0):
        get_ipython().system('mv $iname false/')
    elif (label == 1):
        get_ipython().system('mv $iname true/')


# ### Build the CNN
# We use the Keras VGG16 application with weights but without top. We add an untrained DNN on top.

# In[ ]:


vgg16 = VGG16(weights='imagenet', include_top=False)

x = vgg16.get_layer('block5_conv3').output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model_final = Model(inputs=vgg16.input, outputs=x)


# Freeze all VGG layers and compile the model.

# In[ ]:


for layer in vgg16.layers:
    layer.trainable = False

model_final.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# ### Setup the Datagenerator 
# The interesting point here is, **that we seem to be able to feed any image size to VGG16 and not only 224x224px**. This is particularly meant to improve accuracy for hard to detect images when invasive plants appear only very small in the image. 

# **Hat tip and thanks to Crequena** for that recommendation. See this thread: https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/comments

# I first trained with a small size of 300/225px until early stopping. Than I trained again with 600/450px until early stopping. Feel free to try with even bigger sizes.

# In[ ]:


# You need to have these three folders each with two subfolders for the two classes.
train_data_dir = "D:/KI/01_keras/_kaggle/_invasiveplants/_train"
validation_data_dir = "D:/KI/01_keras/_kaggle/_invasiveplants/_validate"
test_data_dir = "D:/KI/01_keras/_kaggle/_invasiveplants/_test"

# 600/450 _ 500/375 _ 400/300 _ 300/225

img_width = 600  # Change image size for training here
img_height = 450 # Change image size for training here

batch_size = 5 # i achieved good and fast results with this small minibatch size for training
batch_size_val = 400 # if Tensorflow throws a memory error while validating at end of epoch, decrease validation batch size her

# set data augmentation parameters here
datagen = ImageDataGenerator(rescale=1., 
    featurewise_center=True,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="reflect")

# normalization neccessary for correct image input to VGG16
datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

# no data augmentation for validation and test set
validgen = ImageDataGenerator(rescale=1., featurewise_center=True)
validgen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)


train_gen = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True, 
        #save_to_dir="_augmented_images/", 
        #save_prefix="aug_"
        )

val_gen = validgen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

test_gen = validgen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode="binary",
        shuffle=False)

train_samples = len(train_gen.filenames)
validation_samples = len(val_gen.filenames)
test_samples = len(test_gen.filenames)


# In[ ]:


now = datetime.now()

# "_tf_logs" is my Tensorboard folder. Change this to your setup if you want to use TB
logdir = "_tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tb = TensorBoard(log_dir=logdir)

epochs=10

# I stopped training automagically with EarlyStopping after 3 consecutive epochs without improvement
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model_final.fit_generator(train_gen, epochs=epochs, 
                          steps_per_epoch=int(train_samples/batch_size), 
                          validation_data=val_gen, 
                          validation_steps=batch_size_val, 
                          verbose=0, callbacks=[early_stopping, tb, TQDMNotebookCallback()])


# After doing two rounds of training until early stopping (one with a small image size, one with a larger size) we do a second round of training that now includes the last convolutional block of the VGG16, that until now was frozen.

# First we printout all layers. Than we freeze all layers up to the last conv block and compile again.

# In[ ]:


for i, layer in enumerate(model_final.layers):
   print(i, layer.name)

for layer in model_final.layers[:15]:
   layer.trainable = False
for layer in model_final.layers[15:]:
   layer.trainable = True


# In[ ]:


model_final.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),  loss='binary_crossentropy', metrics=['accuracy'])


# Again I did two rounds of training in this second step: First round with a small image size until early stopping, than a second round with the large image size.

# In[ ]:


now = datetime.now()

# "_tf_logs" is my Tensorboard folder. Change this to your setup if you want to use TB
logdir = "_tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tb = TensorBoard(log_dir=logdir)

epochs=50

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model_final.fit_generator(train_gen, epochs=epochs, 
                          steps_per_epoch=int(train_samples/batch_size), 
                          validation_data=val_gen, 
                          validation_steps=int(validation_samples/batch_size), 
                          verbose=0, callbacks=[early_stopping, tb, TQDMNotebookCallback()])


# Make predictions for test images and save as submission CSV.

# In[ ]:


preds = model_final.predict_generator(test_gen, 1531)
preds_rounded = []

for pred in preds:
    if (pred > .5):
        preds_rounded.append("1")
    else:
        preds_rounded.append("0")

preds_filenames = [int(x.replace("test\\", "").replace(".jpg", "")) for x in test_gen.filenames]   

data = (list(zip(preds_filenames, preds_rounded)))

df_result = pd.DataFrame(data, columns=["name", "invasive"])
df_result = df_result.sort_values("name")
df_result.index = df_result["name"]
df_result = df_result.drop(["name"], axis=1)

df_result.to_csv("_kaggle/_invasiveplants/submission_03.csv", encoding="utf8", index=True)


# In[ ]:




