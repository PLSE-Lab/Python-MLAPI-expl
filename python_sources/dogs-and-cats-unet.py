#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import os, shutil
print(len(os.listdir('/kaggle/input/the-oxfordiiit-pet-dataset/images/images')))
len(os.listdir('/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps'))


# In[ ]:


import matplotlib.pyplot as plt
image =  plt.imread('/kaggle/input/the-oxfordiiit-pet-dataset/images/images/Bombay_144.jpg')
plt.imshow(image)
plt.show()
image.shape
image1 =  plt.imread('/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps/Bombay_144.png')
plt.imshow(image1)
plt.show()
image1.shape


# In[ ]:



shutil.rmtree('train', ignore_errors=True) # for deleting directories
shutil.rmtree('val',ignore_errors=True)

os.mkdir('train')
os.mkdir('train/masks')
os.mkdir('train/images')
os.mkdir('val')
os.mkdir('val/masks')
os.mkdir('val/images')

print(os.listdir('train'))
os.listdir('val')


# In[ ]:


import random as rn

src = '/kaggle/input/the-oxfordiiit-pet-dataset/images/images/'
src1 = '/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps/'
train_mask = 'train/masks'
train_img = 'train/images'
val_img = 'val/images'
val_mask = 'val/masks'

files = rn.sample(os.listdir(src), 1024+256)


for f in files[:1024]:
    shutil.copy2(src+f, train_img)
    shutil.copy2(src1+f[:-3]+'png', train_mask)

# val_file = rn.sample(os.listdir('images/images'), 256)


for f in files[1024:]:
    shutil.copy2(src+f, val_img)
    shutil.copy2(src1+f[:-3]+'png', val_mask)

print(len(os.listdir('train/images')))
print(len(os.listdir('train/masks')))
print(len(os.listdir('val/masks')))
print(len(os.listdir('val/images')))


# In[ ]:


for i in os.listdir('train/images'):
    if '.jpg' not in i:
        print(i)
        break
        
for i in os.listdir('val/images'):
    if '.jpg' not in i:
        print(i)
        break


# In[ ]:


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        # new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0]*new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    return (img/255, mask)



def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = 'rgb',
                    mask_color_mode = "rgb", flag_multi_class = False, num_class = 2, save_to_dir = None, 
                   target_size = (256, 256), seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict )
    mask_datagen = ImageDataGenerator(**aug_dict )
    
    image_generator = image_datagen.flow_from_directory(train_path, classes = [image_folder], class_mode = None, 
                                                        color_mode = image_color_mode, target_size = target_size, 
                                                        batch_size = batch_size, seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(train_path, classes = [mask_folder], class_mode = None, 
                                                      color_mode = mask_color_mode, target_size = target_size, 
                                                      batch_size = batch_size, seed = seed )
    

    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# In[ ]:


import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights = None, input_size = (256, 256, 3)):
    inputs = Input(input_size)
    # print(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #encoder part over
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    #decoder part starts
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3 )
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal' )(merge9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv10 = Conv2D(3, 1, activation = 'sigmoid' )(conv9)

    model = Model(input = inputs, output = conv9)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = ['accuracy'])
    # dice_coef_loss
    # model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# In[ ]:


model = unet()


# In[ ]:


data_gen = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, 
                     zoom_range=0.05, horizontal_flip=True, fill_mode='nearest' )

train_gen = trainGenerator(32, 'train', 'images', 'masks', data_gen, flag_multi_class = True, 
                           num_class = 3)

val_gen = trainGenerator(8, 'val', 'images', 'masks', dict(), flag_multi_class = True, num_class = 3)


# In[ ]:


shutil.rmtree('unet_membrane.hdf5',ignore_errors=True)


# In[ ]:


model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True )

model_hist = model.fit_generator(train_gen, steps_per_epoch=1024//32, epochs=25, validation_data = val_gen, 
                    validation_steps = 256//8, callbacks=[model_checkpoint])


# In[ ]:


image = io.imread(src1+'yorkshire_terrier_12.png')
plt.imshow(image)


# In[ ]:


target_size = (256,256)
flag_multi_class = False
img = io.imread(src+'yorkshire_terrier_22.jpg')
img = img / 255
img = trans.resize(img, target_size)
img = np.reshape(img, (1,)+img.shape) 
# print(img.shape)
rez = model.predict(img)
# print(rez.shape)
plt.imshow(rez[0,:,:,:])
plt.show()


# In[ ]:


def display(display_list):
    plt.figure(figsize=(10, 10))
    # print(display_list.shape)
    title = ['Input','True Mask','Predicted raw Mask','Predicted Mask']
#     print(len(display_list))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i==1 and len(display_list) == 4:
            plt.imshow(display_list[1])
        else:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         if i==3 and len(display_list) > 3:
#             plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         else:
#             plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
#     print(pred_mask.shape)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

display([create_mask(rez)])


# In[ ]:


loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']
acc = model_hist.history['accuracy']
val_acc = model_hist.history['val_accuracy']
epochs = range(25)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
# plt.ylim([0, 1])
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy Value')
# plt.ylim([0, 1])
plt.legend()
plt.show()


# In[ ]:


test_file = rn.sample(os.listdir(src), 10)


# In[ ]:


for i in test_file:
    img = io.imread(src+i)
    mask = io.imread(src1+i[:-3]+'png')
#     print(mask.shape)
    img = img / 255
    img = trans.resize(img,target_size)
    img = np.reshape(img,(1,)+img.shape)
    show = [img[0,:, :,:], mask, model.predict(img)[0,:,:,:], create_mask(model.predict(img))]
    display(show)
#     model.predict(img)[0,:,:,:]

