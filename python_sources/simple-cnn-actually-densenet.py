#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm # progress bar
import gc # garbage collector
import random
import cv2
import albumentations as A

# CNN
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,Input,UpSampling2D,concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import DenseNet121

# Training
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator # super useful because we don't need to create our own generator
from sklearn.model_selection import train_test_split # train and test data split

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


IMG_SIZE=64
N_CHANNELS=1


# In[ ]:


# https://www.kaggle.com/shawon10/bangla-graphemes-image-processing-deep-cnn
    # Images can lose much important information due to resizing. Inter-area interpolation 
    # is preferred method for image decimation. Interpolation works by using known data to 
    # estimate values at unknown points. We use inter-area interpolation after resizing images. 
    # We have also cropped the images and extract the region of interest of the images.
    # Cropped images are performing better.

def resize(df, size=IMG_SIZE, need_progress_bar=True):
    resized = {}
    resize_size=IMG_SIZE
    angle=0
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(137,236)
            #Centering
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            #Scaling
            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            #Removing Blur
            #aug = A.GaussianBlur(p=1.0)
            #image = aug(image=image)['image']
            #Noise Removing
            #augNoise=A.MultiplicativeNoise(p=1.0)
            #image = augNoise(image=image)['image']
            #Removing Distortion
            #augDist=A.ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=1.0)
            #image = augDist(image=image)['image']
            #Brightness
            augBright=A.RandomBrightnessContrast(p=1.0)
            image = augBright(image=image)['image']
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            #image=affine_image(image)
            #image= crop_resize(image)
            #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
            #image=resize_image(image,(64,64))
            #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
            #gaussian_3 = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT) #unblur
            #image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
            #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
            #image = cv2.filter2D(image, -1, kernel)
            #ret,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(137,236)
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            #Removing Blur
            #aug = A.GaussianBlur(p=1.0)
            #image = aug(image=image)['image']
            #Noise Removing
            #augNoise=A.MultiplicativeNoise(p=1.0)
            #image = augNoise(image=image)['image']
            #Removing Distortion
            #augDist=A.ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=1.0)
            #image = augDist(image=image)['image']
            #Brightness
            augBright=A.RandomBrightnessContrast(p=1.0)
            image = augBright(image=image)['image']
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            #image=affine_image(image)
            #image= crop_resize(image)
            #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
            #image=resize_image(image,(64,64))
            #image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
            #gaussian_3 = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT) #unblur
            #image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
            #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
            #image = cv2.filter2D(image, -1, kernel)
            #ret,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized


# In[ ]:


# kaggle.com/xhlulu/bengali-ai-simple-densenet-in-keras
def build_model(densenet):
    x_in = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(3, (3, 3), padding='same')(x_in)
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='dense_5')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    head_root = Dense(168, activation='softmax',name='dense_1')(x)
    head_vowel = Dense(11, activation='softmax',name='dense_2')(x)
    head_consonant = Dense(7, activation='softmax',name='dense_3')(x)
    
    model = Model(inputs=x_in, outputs=[head_root, head_vowel, head_consonant])
    
    model.compile(
        Adam(lr=0.0001), 
        metrics=['accuracy'], 
        loss='categorical_crossentropy'
    )
    
    return model


# In[ ]:


weights_path = '/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
densenet = DenseNet121(include_top=False, weights=weights_path, input_shape=(IMG_SIZE, IMG_SIZE, 3))


# In[ ]:


model = build_model(densenet)


# In[ ]:


model.summary()


# In[ ]:


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Reduce learning rate when a metric has stopped improving
# if validation loss of dense_x is not decreasing for 3 consecutive epochs, decrease the learning rate by 0.5 
# given if learning rate is above 0.00001 and give us little insight on what has happened verbose=1
ReduceLearningOnPlateau_root = ReduceLROnPlateau(monitor='val_dense_1_loss',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
ReduceLearningOnPlateau_vowel = ReduceLROnPlateau(monitor='val_dense_2_loss',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
ReduceLearningOnPlateau_consonant = ReduceLROnPlateau(monitor='val_dense_3_loss',patience=3,verbose=1,factor=0.5,min_lr=0.00001)


# Stop training when a monitored quantity has stopped improving
EarlyStopping = EarlyStopping(monitor='val_loss',patience=4, min_delta=0.0025,restore_best_weights=True)
# stop the model from fitting data if validation loss has not decreased by 0.0025 in the last 5 epochs and 
# restore best weights for the next time
# loss vs val_loss: 
    # if loss is significantly lower than val_loss the data is probably overfitted
    # EarlyStopping prevents this from happening (from overfitting)


# Save the model after every epoch
ModelCheckpoint = ModelCheckpoint('best_model_weight.h5', monitor='val_loss', verbose=1, 
                      save_best_only=True, save_weights_only=True)
# save the weights in a file name specified only if the validation loss of out_1 layer has improved from 
# last save. out_1 because it's recall matters twice 

# save all into one array
callbacks = [ReduceLearningOnPlateau_root, ReduceLearningOnPlateau_vowel, ReduceLearningOnPlateau_consonant, EarlyStopping, ModelCheckpoint]


# In[ ]:


batch_size = 128
epochs = 30


# In[ ]:


num_dataset = 4


# In[ ]:


# Function takes data and label arrays and then generates batches of augmented data
# Many notebooks are using exactly this kind of generator
# the only different thing is usually different batch size
class TweakedDataGenerator(ImageDataGenerator):

    def flow(self, x, y=None, batch_size=batch_size, shuffle=True, sample_weight=None, seed=None, 
             save_to_dir=None, save_prefix='', save_format='png', subset=None):

        labels_array = None # concatenate all labels arrays into a single array
        target_lengths = {} # dictionary mapping the 'key' (y1,y2,...) to lengths of corresponding label_array     
        ordered_labels = [] # store ordering in which the labels were passed to this class
        
        for output, label_value in y.items():
            if labels_array is None:
                labels_array = label_value # for the first time loop, it's empty, so insert first element
            else:
                labels_array = np.concatenate((labels_array, label_value), axis=1)
            target_lengths[output] = label_value.shape[1]
            ordered_labels.append(output)


        for flowx, flowy in super().flow(x, labels_array, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_labels:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict


# In[ ]:


history_list = []
train_data = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

for i in range(num_dataset):

    # Merge data frames, then drop useless columns
    b_train_data =  pd.merge(pd.read_parquet(f'../input/bengaliai-cv19/train_image_data_{i}.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)
    X_train = b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis=1)
    
    # Normalize
    X_train = resize(X_train)/255

    
    # Reshape, -1 means we let numpy figure out the dimensions
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
  

    # Training set for the three consitutuents
    Y_train_root = b_train_data['grapheme_root']
    Y_train_vowel = b_train_data['vowel_diacritic']
    Y_train_consonant = b_train_data['consonant_diacritic']

    
    # Extract labels
    # Convert numerical values of classes into dummy variables
    Y_train_root = pd.get_dummies(Y_train_root).values
    Y_train_vowel = pd.get_dummies(Y_train_vowel).values
    Y_train_consonant = pd.get_dummies(Y_train_consonant).values
    
    #print(f'Training images: {X_train.shape}')
    # print(f'Training labels root: {Y_train_root.shape}')
    # print(f'Training labels vowel: {Y_train_vowel.shape}')
    # print(f'Training labels consonants: {Y_train_consonant.shape}')

    
    # Split the data in training and test to cross-validate our model's performance
    # random_state produces the same data every time for reproducibility so it is kind of pseudo randomness
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=2020)

    # Data augmentation for creating more training data
    datagen = TweakedDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False # randomly flip images
        ) 

    datagen.fit(x_train)   
    print(y_train_root.shape)
    # Start training
    history = model.fit_generator(datagen.flow(x_train, {'dense_1': y_train_root, 'dense_2': y_train_vowel, 'dense_3': y_train_consonant}, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                              steps_per_epoch=x_train.shape[0] // batch_size, callbacks = callbacks)
    
    stopped_at = EarlyStopping.stopped_epoch # gives you >=1 at which epoch model stopped due to early stopping
    
    history_list.append(history)
    
    # Delete to reduce memory usage
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant   
    del b_train_data
    del X_train
    del Y_train_root, Y_train_vowel, Y_train_consonant
    
    # Use this method to force the system to try to reclaim the maximum amount of available memory
    gc.collect()
    
del train_data
gc.collect()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def plot_loss(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    epoch = len(his.history['loss'])
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_1_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_2_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_3_loss'], label='train_consonant_loss')
    
    plt.plot(np.arange(0, epoch), his.history['val_dense_1_loss'], label='val_train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_dense_2_loss'], label='val_train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_dense_3_loss'], label='val_train_consonant_loss')
    
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    epoch = len(his.history['dense_1_accuracy'])
    plt.plot(np.arange(0, epoch), his.history['dense_1_accuracy'], label='train_root_acc')
    plt.plot(np.arange(0, epoch), his.history['dense_2_accuracy'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['dense_3_accuracy'], label='train_consonant_accuracy')
    
    plt.plot(np.arange(0, epoch), his.history['val_dense_1_accuracy'], label='val_root_acc')
    plt.plot(np.arange(0, epoch), his.history['val_dense_2_accuracy'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_dense_3_accuracy'], label='val_consonant_accuracy')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()


# In[ ]:


for dataset in range(num_dataset):
    plot_loss(history_list[dataset], epochs, f'Training Dataset: {dataset}')
    plot_acc(history_list[dataset], epochs, f'Training Dataset: {dataset}')


# In[ ]:


# Load the best weights so far so that we don't have over fitting by the time the mode lhad stopped 
model.load_weights('best_model_weight.h5') 


# In[ ]:


preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}


# In[ ]:


components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[]
row_id=[]

for i in range(num_dataset):
    df_test_img = pd.read_parquet('../input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 
    df_test_img.set_index('image_id', inplace=True)

    X_test = resize(df_test_img, need_progress_bar=False)/255
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    preds = model.predict(X_test)

    for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[i], axis=1)

    for k,id in enumerate(df_test_img.index.values):  
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])
    del df_test_img
    del X_test
    gc.collect()
    
submission = pd.DataFrame({'row_id': row_id, 'target':target}, columns = ['row_id','target'])
submission.to_csv('submission.csv',index=False)
submission.head()


# In[ ]:


train_data = pd.read_csv('../input/bengaliai-cv19/train.csv')

b_train_data =  pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)
X_train = b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)
X_train = resize(X_train)/255

X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)


# ![](http://)

# vanaf hier origineel
# 
