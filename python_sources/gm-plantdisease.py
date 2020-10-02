#!/usr/bin/env python
# coding: utf-8

# In this kernel, I will be dealing with Plant Disease Dataset. The dataset has been taken from SP Mohanty's repository. The data set has ~20000 images belonging to 15 classes from 3 crops. 

# In[ ]:


import random 
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import metrics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


from keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



SEED=1
os.listdir()


# Since I will be using ImageGenerator, a class wise folder generation will be needed. So I have used os library as well as shell scripting has been used to split samples into validation and training set in 20-80 split ratio. 
# 
# The dataset had 15 subfolders each containing leaf images corresponding to the labels given by their folder name.

# In[ ]:


#Checking length of each class type type

os.chdir('../input/plantdisease/plantvillage/PlantVillage/')
print(len(os.listdir('Pepper__bell___Bacterial_spot')))
print(len(os.listdir('Pepper__bell___healthy')))
print(len(os.listdir('Potato___Early_blight')))
print(len(os.listdir('Potato___Late_blight')))
print(len(os.listdir('Potato___healthy')))
print(len(os.listdir('Tomato_Bacterial_spot')))
print(len(os.listdir('Tomato_Early_blight')))
print(len(os.listdir('Tomato_Late_blight')))
print(len(os.listdir('Tomato_Leaf_Mold')))
print(len(os.listdir('Tomato_Septoria_leaf_spot')))
print(len(os.listdir('Tomato_Spider_mites_Two_spotted_spider_mite')))
print(len(os.listdir('Tomato__Target_Spot')))
print(len(os.listdir('Tomato__Tomato_YellowLeaf__Curl_Virus')))
print(len(os.listdir('Tomato__Tomato_mosaic_virus')))
print(len(os.listdir('Tomato_healthy')))


# Since the dataset which has been imported is read-only, we need to make a copy in our /kaggle/input/ directory

# In[ ]:


#Copying the dataset which is read-only

os.system('scp -r /kaggle/input/plantdisease/plantvillage/PlantVillage /kaggle/input/')


# In[ ]:


#Just checking the files in the library
get_ipython().system('ls -lrt')


# # Making directories from the dataset so that ImageGenerator can be used
# 
# We have to split the sub-directories into validation and training set, so creating 

# In[ ]:


base_dir='/kaggle/input/PlantVillage/'

#Creating training and validation directory 
os.chdir(base_dir)

os.mkdir('dataset')
os.mkdir('training')
os.mkdir('validation')

os.chdir('dataset')

os.chdir(base_dir)
os.listdir()
os.chdir('dataset')
os.system('scp -r ../Tom* ../Pepper* ../Potato* .')


# In[ ]:


#Creating subdirectories in each of the two folders viz training and validation
classes = os.listdir()
for i in classes:
    tr_dir = os.path.join('/kaggle/input/PlantVillage/training',i)
    val_dir = os.path.join('/kaggle/input/PlantVillage/validation',i)
    
    os.mkdir(tr_dir)
    os.mkdir(val_dir)
    
    
# Checking

print(os.listdir('/kaggle/input/PlantVillage/training'))

print(os.listdir('/kaggle/input/PlantVillage/validation'))


# # Splitting Data into Training and Validation Set in 80-20 ratio
# 
# Splitting the data into 80-20 ratio across each of folders (Validation and Training)
# 
# **We need to make sure that the division of the data, while being random, should be stratified across classes. Which means we need to make sure that the division is uniform across classes as well, the loop below makes sure that the division is stratified and random(within the folders)**

# In[ ]:


print(classes)

print(os.getcwd())

for item in classes:
    n_val = round(len(os.listdir(item))*.2)
    n_train = round(len(os.listdir(item))*.8)
    fnames = os.listdir(item)
    
    assert(n_val+n_train == len(fnames))
    
    random.seed(SEED+5)
    random.shuffle(fnames)
    val_fnames = fnames[0:n_val]
    tr_fnames = fnames[n_val:len(fnames)]
    
    assert(len(val_fnames)+len(tr_fnames)==len(fnames))
    
    for i in val_fnames:
        src ='/kaggle/input/PlantVillage/dataset/{}/{}'.format(item,i)
        dest = '/kaggle/input/PlantVillage/validation/{}/'.format(item)
        shutil.copy(src,dest)
        
    for j in tr_fnames:
        src ='/kaggle/input/PlantVillage/dataset/{}/{}'.format(item,j)
        dest = '/kaggle/input/PlantVillage/training/{}/'.format(item)
        shutil.copy(src,dest)      


# Counting number of samples across each class

# In[ ]:


for i in classes:
    path=os.path.join('/kaggle/input/PlantVillage/training/',i)
    print('Training samples in {} is {}'.format(i,len(os.listdir(path))))
    
    path=os.path.join('/kaggle/input/PlantVillage/validation/',i)
    print('Validation samples in {} is {}\n'.format(i,len(os.listdir(path))))
    
##Removing class files from the base directory
os.system('rm -r /kaggle/input/PlantVillage/Tom* /kaggle/input/PlantVillage/Pepper* /kaggle/input/PlantVillage/Potato* .')


# In[ ]:


validation_dir = '/kaggle/input/PlantVillage/validation/'
training_dir = '/kaggle/input/PlantVillage/training/'


# # Architecture
# 
# Using Keras Library to build our architecture
# 
# 
# The model has been updated over each commit.
# 
# v9
# Filter size has been changed from 3x3 to 2x2.
# 
# v10
# Changed filter size to 3x3 on top layers, whereas 2x2 on lower layers to learn smaller abstractions.
# 
# v11
# Changed n_epochs to 100. Increased learning rate to 1e-3
# 
# NB: **The model has used 3 callbacks**
# 
# * EarlyStopping
# * Decrease Learning Rate if the learning hits a plateau
# * 
# 
# 

# In[ ]:


#Simple Deep Learning Architecture

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_categorical_loss')>0.97):
            print("\nReached 97% validation accuracy so cancelling training!")
            self.model.stop_training = True



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(axis=-1),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=-1),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=-1),
    
    tf.keras.layers.Conv2D(256, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(512, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3,decay=1e-4/500),
              metrics=['categorical_accuracy'])#,top_3_categ_acc,top_5_categ_acc])

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=90,
      width_shift_range=0.5,
      height_shift_range=0.5,
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=True,
      #vertical_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 100 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(256, 256),  # All images will be resized to 256x256
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')


model_path = '/kaggle/working/bst_mdl_plant_disease.hdf5'

callbacks = myCallback()
earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max', restore_best_weights=True)
modelcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_categorical_accuracy', mode='max')
reduce_lr_loss_on_plateau = ReduceLROnPlateau(monitor='val_categorical_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')

callback_lists = [earlyStopping, modelcp_save, reduce_lr_loss_on_plateau ]

history = model.fit_generator(
      train_generator,
      steps_per_epoch=500,  # 16511 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=128,  # 4127 images = batch_size * steps
      verbose=1,
      callbacks = callback_lists
      )


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.savefig('/kaggle/working/Acc_plot.png',dpi=200)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

plt.savefig('/kaggle/working/Loss_plot.png',dpi=200)


acc_l= pd.DataFrame(acc)
val_acc_l=pd.DataFrame(val_acc)

loss_df = pd.DataFrame(loss)
val_loss_df = pd.DataFrame(val_loss)

acc_l.to_csv('/kaggle/working/accuracy.csv')
val_acc_l.to_csv('/kaggle/working/val_acc.csv')

loss_df.to_csv('/kaggle/working/loss.csv')
val_loss_df.to_csv('/kaggle/working/val_loss.csv')


# [](http://)* * Using Inception Net to achieve better accuracy

# In[ ]:


'''from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras import applications


# Downloading the pretrained model


from tensorflow.keras.applications.inception_v3 import InceptionV3


#Initialising the model with inbuilt InceptionV3
pre_trained_model = InceptionV3(input_shape = (256, 256, 3), 
                                weights='imagenet',
                                include_top = False, 
                                )


#Setting the layers as not trainable, as we want it to use the pretrained weights during epochs
for layer in pre_trained_model.layers:
  layer.trainable = False
  

pre_trained_model.summary()

#Pointing towards a stage in the model (refer Summary), we are telling that last_layer will be the 'mixed6' layer, similarly we assign the last output which need not be trained
last_layer = pre_trained_model.get_layer('mixed6') #Tried mixed7
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output'''


# In[ ]:


'''## Adding our paramaters to the model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc')>0.97):
            print("\nReached 97% validation accuracy so cancelling training!")
            self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers import Adam
# Flatten the output layer to 1 dimension

x = layers.GlobalAveragePooling2D()(last_output)

x = layers.Dropout(0.5)(x) 

x = layers.Conv2D(256,(3,3),activation='relu')(x)
x = layers.layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPooling2D(2, 2)(x)


x = layers.GlobalAveragePooling2D()(x)
x = layers.Conv2D(128,(3,3),activation='relu')(x)
x = layers.layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPooling2D(2, 2)(x)


x = layers.GlobalAveragePooling2D()(x)
x = layers.Conv2D(64,(3,3),activation='relu')(x)
x = layers.layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Flatten()(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dense(1024, activation='relu')(x)

x = layers.Dense(512, activation='relu')(x)

x = layers.Dropout(0.5)(x)                  
# Add a final softmax layer for classification into 15 categories
x = layers.Dense(15, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

def top_3_categ_acc(y_pred,y_true):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_categ_acc(y_pred,y_true):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])#,top_3_categ_acc,top_5_categ_acc])

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=70,
      width_shift_range=0.5,
      height_shift_range=0.5,
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=True,
      #vertical_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 100 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(256, 256),  # All images will be resized to 256x256
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

callbacks = myCallback()

history = model.fit_generator(
      train_generator,
      steps_per_epoch=258,  # 16512 images = batch_size * steps
      epochs=50,
      validation_data=validation_generator,
      validation_steps=64,  # 4126 images = batch_size * steps
      verbose=1,
      callbacks = [callbacks])'''


# In[ ]:


'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

acc_l= pd.DataFrame(acc)
val_acc_l=pd.DataFrame(val_acc)

acc_l.to_csv('accuracy.csv')
val_acc_l.to_csv('val_acc.csv')


model.save('DeepLearn.h5')
model.save_weights('DeepLearn_Weights.h5')
'''

