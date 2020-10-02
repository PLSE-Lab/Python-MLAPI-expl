#!/usr/bin/env python
# coding: utf-8

# # Skin Lesion Classification
# ## Transfer Learning using *MobileNetV2* model

# ## Prepare Dataset

# In[ ]:


import os
print(os.listdir("../input/skin-cancer-mnist-ham10000"))


# In[ ]:


# Dataset directory
data_dir = "../input/skin-cancer-mnist-ham10000"
base_dir = "../input/ham10000"
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
val_dir   = os.path.join(base_dir, 'val')
os.mkdir(val_dir)
print(os.listdir(base_dir))


# In[ ]:


# Create new folders in the training directory for each of the classes
nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)


# In[ ]:


# Create new folders in the validation directory for each of the classes
nv = os.path.join(val_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)


# In[ ]:


# Read the metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

df = pd.read_csv(data_dir +'/HAM10000_metadata.csv')
df.head()


# In[ ]:


# Set y as the labels
y = df['dx']
# Split to train and validation set
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


# In[ ]:


# Find the number of values in the training and validation set
df_train['dx'].value_counts()
df_val['dx'].value_counts()

# Transfer the images into folders
# Set the image id as the index
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir(data_dir +'/ham10000_images_part_1')
folder_2 = os.listdir(data_dir +'/ham10000_images_part_2')


# In[ ]:


# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])


# In[ ]:


# copy the training images to train_dir
for image in train_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join(data_dir+'/ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join(data_dir+'/ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)


# In[ ]:


# show how many training images are in each folder
print(len(os.listdir(train_dir+'/nv')))
print(len(os.listdir(train_dir+'/mel')))
print(len(os.listdir(train_dir+'/bkl')))
print(len(os.listdir(train_dir+'/bcc')))
print(len(os.listdir(train_dir+'/akiec')))
print(len(os.listdir(train_dir+'/vasc')))
print(len(os.listdir(train_dir+'/df')))


# In[ ]:


# copy the validation images val_dir
for image in val_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join(data_dir+'/ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join(data_dir+'/ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)


# In[ ]:


# show how many validation images are in each folder
print(len(os.listdir(val_dir+'/nv')))
print(len(os.listdir(val_dir+'/mel')))
print(len(os.listdir(val_dir+'/bkl')))
print(len(os.listdir(val_dir+'/bcc')))
print(len(os.listdir(val_dir+'/akiec')))
print(len(os.listdir(val_dir+'/vasc')))
print(len(os.listdir(val_dir+'/df')))


# In[ ]:


# Import Libraries
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


# ## Prepare Dataset

# In[ ]:


batch_size = 32
# Data Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(train_dir,
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator=val_datagen.flow_from_directory(val_dir,
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


# In[ ]:


num_classes = 7
prediction_labels = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


# ## Build Model

# In[ ]:


# Load Model (keras built-in model)
base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.resnet50.ResNet50(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.densenet.DenseNet121(input_shape=(224,224,3), weights='imagenet',include_top=False)

# Add Extra Layers to Model
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(64,activation='relu')(x) 
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


# In[ ]:


# Check layers no. & name
for i,layer in enumerate(model.layers):
    print(i,layer.name)


# ### MobileNetV2 : *layer 155*, ResNet50 : *layer 175*, DenseNet121 : *layer 427*

# In[ ]:


# set extra layers to trainable 
for layer in model.layers[:155]:
    layer.trainable=False
for layer in model.layers[155:]:
    layer.trainable=True


# In[ ]:


# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


# Add weights to make the model more sensitive to melanoma
class_weights={
    0: 1.0,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 1.0,  # df
    4: 3.0,  # mel
    5: 1.0,  # nv
    6: 1.0,  # vasc
}


# In[ ]:


# Train Model (target is loss <0.01)
num_epochs = 20
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
model.fit_generator(train_generator, steps_per_epoch=step_size_train, epochs=num_epochs, class_weight=class_weights, validation_data=val_generator, validation_steps=step_size_val)


# ## Save Model

# In[ ]:


# Save Model
model.save('tl_skinlesion.h5')


# ## Evaluate Model

# In[ ]:


# Evaluate Model
loss, acc = model.evaluate_generator(val_generator, steps=step_size_val)
print("The accuracy of the model is {:.3f}\nThe Loss in the model is {:.3f}".format(acc,loss))


# In[ ]:




