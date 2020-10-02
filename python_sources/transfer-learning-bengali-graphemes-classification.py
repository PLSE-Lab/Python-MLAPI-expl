#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from tqdm.auto import tqdm
import time, gc

import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', None)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import albumentations as A
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, Concatenate
from keras.layers import MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.initializers import RandomNormal
from keras.applications import DenseNet121

from sklearn.model_selection import train_test_split

start_time = time.time()


# ## Resource path setting 

# In[ ]:


dataset = '/kaggle/input/bengaliai-cv19'
pretrained = '../input/bangla-graphemes-pretrained-weights'


# ## Checking Model

# In[ ]:


if os.path.isfile(os.path.join(pretrained,"GraphemeDenseNet121.h5"))         and os.path.isfile(os.path.join(pretrained,"hist.csv")):
    print('Model is present')
else:
    print("Error. No Model Found")


# ## Size and Channel of images

# In[ ]:


SIZE = 100   # input image size
N_ch = 1


# ##  Loading Pretrained Densenet121 Model
# ### Batch Size: 256
# ### Epochs: 30 (Early Stopped in 20)

# In[ ]:


model = load_model(os.path.join(pretrained, 'GraphemeDenseNet121.h5'))


# ## DenseNet121 Model Summary

# In[ ]:


model.summary()


# ## Loading Images and Pre-processing

# In[ ]:


# Resize image size
def resize(df, size=100):
    resized = {}
    resize_size=100
    angle=0
    for i in range(df.shape[0]):
            image=df.loc[df.index[i]].values.reshape(137,236)
            #Centering
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            #Scaling
            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_AREA,
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
    resized_df = pd.DataFrame(resized).T
    return resized_df


# ## Accuracy and Loss Curve

# In[ ]:


df = pd.read_csv(os.path.join(pretrained,'hist.csv'))
    
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(1, 2, figsize = (12, 4))

ax[0].plot(df[['root_loss','vowel_loss','consonant_loss',
               'val_root_loss','val_vowel_loss','val_consonant_loss']])
ax[0].set_ylim(0, 2)
ax[0].set_title('Loss')
ax[0].legend(['train_root_loss','train_vowel_loss','train_conso_loss',
              'val_root_loss','val_vowel_loss','val_conso_loss'],
             loc='upper right')
ax[0].grid()
ax[1].plot(df[['root_acc','vowel_acc','consonant_acc',
               'val_root_acc','val_vowel_acc','val_consonant_acc']])
ax[1].set_ylim(0.5, 1)
ax[1].set_title('Accuracy')
ax[1].legend(['train_root_acc','train_vowel_acc','train_conso_acc',
              'val_root_acc','val_vowel_acc','val_conso_acc'],
             loc='lower right')
ax[1].grid()


# ## Target Columns

# In[ ]:


tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']


# ## Prediction on Test Images

# In[ ]:


row_ids = []
targets = []      
id = 0
for i in range(4):
    img_df = pd.read_parquet(os.path.join(
                            dataset, 'test_image_data_'+str(i)+'.parquet'))
    img_df = img_df.drop('image_id', axis = 1)
    img_df = resize(img_df, SIZE) / 255
    X_test = img_df.values.reshape(-1, SIZE, SIZE, N_ch)

    preds = model.predict(X_test)
    for j in range(len(X_test)):
        for k in range(3):
            row_ids.append('Test_'+str(id)+'_'+tgt_cols[k])
            targets.append(np.argmax(preds[k][j]))
        id += 1


# ## Creating Submission CSV File

# In[ ]:


df_submit = pd.DataFrame({'row_id':row_ids,'target':targets},
                         columns = ['row_id','target'])
df_submit.to_csv('submission.csv',index=False)
df_submit.head(10)

