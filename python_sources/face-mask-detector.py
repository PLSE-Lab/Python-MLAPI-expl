#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D


# In[ ]:



filename = "../input/face-mask-detection-dataset/train.csv"
data = pd.read_csv(filename)
data.head()


# In[ ]:


data['classname'].unique()


# In[ ]:



data['cat']=np.where((data['classname']=='face_with_mask') | (data['classname']=='mask_colorful') | (data['classname']=='face_with_mask_incorrect') | (data['classname']=='mask_surgical') | (data['classname']=='face_other_covering') | (data['classname']=='scarf_bandana') | (data['classname']=='hijab_niqab') | (data['classname']=='gas_mask') | (data['classname']=='balaclava_ski_mask'),'mask','no_mask')


# In[ ]:


data.head()


# In[ ]:


data.drop(['classname','x1','x2','y1','y2'],axis=1,inplace=True)


# In[ ]:


data.values


# In[ ]:


ls


# In[ ]:


annotations_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/'
images_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'


# In[ ]:


import os
from PIL import Image
import xml.etree.ElementTree as ET
import sys

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


rootdir = os.getcwd()
CroppedFolder = "Cropped"
Annotation = "Annotation"
if not os.path.exists(CroppedFolder):
    os.makedirs(CroppedFolder)
    print 'Created "/Cropped/" directory'

for directory in get_immediate_subdirectories(rootdir):	#loop over all dirs
		if directory != CroppedFolder and directory != Annotation:
			print "Processing Directory: " + directory
			if not os.path.exists(os.path.join(CroppedFolder, directory)):
    				print 'Created "/Cropped/' + directory + '" directory'
    				os.makedirs(os.path.join(CroppedFolder, directory))
			for filename in os.listdir(directory):
				basename =  os.path.splitext(filename)[0]
				try:
					file = open ( os.path.join( Annotation, directory, basename))
					root = ET.fromstring(file.read())
					file.close()
					xmin = int (root.find('object').find('bndbox').find('xmin').text)
					ymin = int (root.find('object').find('bndbox').find('ymin').text)
					xmax = int (root.find('object').find('bndbox').find('xmax').text)
					ymax = int (root.find('object').find('bndbox').find('ymax').text)
					img =  Image.open( os.path.join(directory, filename) )
					cropped = img.crop((xmin, ymin, xmax, ymax))
					save_file = open (os.path.join(CroppedFolder, directory, filename), 'w')
					cropped.save(os.path.join(CroppedFolder, directory, filename), "JPEG")
					save_file.close()


				except Exception, e:
					print "Exception encountered at basename " + basename + " with path as " +  os.path.join( Annotation, directory, basename) 
					print "Unexpected error:", str(e)


# In[ ]:


train_dir = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
train_sep_dir = 'train/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

for filename, class_name in data.values:
    src_path = train_dir + filename
    dst_path = train_sep_dir + class_name + '/' + filename
    mask_path= train_sep_dir + class_name
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    shutil.copy(src_path,dst_path)


# In[ ]:



img=cv2.imread('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/0001.jpg')
img.shape


# In[ ]:



batch_size = 64
IMG_HEIGHT = 256
IMG_WIDTH = 256


# In[ ]:


image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.3)    

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory='train',
                                                 shuffle=True,
                                                 target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                                 subset="training",
                                                 class_mode='binary')


# In[ ]:



validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory='train',
                                                 shuffle=True,
                                                 target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                                 subset="validation",
                                                 class_mode='binary')


# In[ ]:



model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:



history=model.fit(
    train_dataset,
    steps_per_epoch = train_dataset.n // train_dataset.batch_size,
    validation_data = validation_dataset, 
    validation_steps = validation_dataset.n // validation_dataset.batch_size,
    epochs = 10)


# In[ ]:


epochs=10
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

