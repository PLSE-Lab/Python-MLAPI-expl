#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import math 
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow.keras import layers,models
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.DataFrame(columns = ['id','class'])#


# In[ ]:


get_ipython().system('rm -rf ../working/pics_all')


# In[ ]:


get_ipython().system('mkdir pics_all')


# In[ ]:


train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')


# In[ ]:


test = np.array(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'))


# In[ ]:


def crop(img):
    img = cv2.resize(img,(img.shape[1]//4,img.shape[0]//4), interpolation =  cv2.INTER_AREA )
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img3 = cv2.threshold(img2,10,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(img2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    if x>5 and h>5:
        img = img[y:y+h,x:x+w,:]
    img = cv2.resize(img,(256,256),interpolation =  cv2.INTER_AREA)
#     cv2.imshow(img)
#     plt.show()
    return img
        


# In[ ]:



def Krish(crop):
    Input=crop[:,:,2]
    a,b=Input.shape
    Kernel=np.zeros((3,3,8))#windows declearations(8 windows)
    Kernel[:,:,0]=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    Kernel[:,:,1]=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    Kernel[:,:,2]=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    Kernel[:,:,3]=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    Kernel[:,:,4]=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    Kernel[:,:,5]=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    Kernel[:,:,6]=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    Kernel[:,:,7]=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    #Kernel=(1/float(15))*Kernel
    #Convolution output
    dst=np.zeros((a,b,8))
    for x in range(0,8):
        dst[:,:,x] = cv2.filter2D(Input,-1,Kernel[:,:,x])
    Out=np.zeros((a,b))
    for y in range(0,a-1):
        for z in range(0,b-1):
            Out[y,z]=max(dst[y,z,:])
    Out=np.uint8(Out)
    return Out


def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def Cos_inv(crop):    
    blu=crop[:,:,0].astype(np.float64)    
    gre=crop[:,:,1].astype(np.float64)    
    red=crop[:,:,2].astype(np.float64)     
    l=((blu**2)+(gre**2)+(red**2))**0.5    
    l=l.astype(np.float64)    
    l[l==0]=0.0000000001    
    m=blu/l    
    Max=np.max(m)    
    Min=np.min(m)    
    j=((m-float(Min)/(float(Max)-float(Min)))*2)-1    
    n=((np.arccos(j))*255)/3.14    
    nm=n.astype(np.uint8)    
    equ1 = cv2.equalizeHist(nm)    
    return equ1


def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img,(img.shape[1]//4,img.shape[0]//4), interpolation = cv2.INTER_AREA)
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = Krish(img)
#     print(img.shape)
    img = cv2.resize(img, (256,256), cv2.INTER_AREA)
#     plt.imshow(img, cmap = 'gray')
#     plt.show()
    #_,img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    return img


# In[ ]:


def transform(img):
    arr = []
#     return arr
    for i in range(4,7,2):
        new_image = np.zeros(img.shape, img.dtype)
#         new_image2 = np.zeros(img.shape, img.dtype)
#         new_image3 = np.zeros(img.shape, img.dtype)
        alpha = i # Simple contrast control
        beta1 = -35*i    # Simple brightness control
#         beta2 = -40*i
#         beta3 = -45*i#
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
        #        new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
                new_image[y,x] = np.clip(alpha*img[y,x] + beta1, 0, 255)
#                 new_image2[y,x] = np.clip(alpha*img[y,x] + beta2, 0, 255)
#                 new_image3[y,x] = np.clip(alpha*img[y,x] + beta3, 0, 255)
        arr.extend([new_image])#,new_image2])#,new_image3])
    return arr


# In[ ]:


import random


# In[ ]:


directory = '../input/aptos2019-blindness-detection/train_images/'
# diag = []
train_arr = []
if test.shape[0]>1928:
    for index,value in train.iterrows():
        id_code = value[0]
        diagnosis = str(value[1])
    #     img = cv2.imread(directory+id_code+'.png')
        img = circle_crop_v2(directory+id_code+'.png')
    #     arr = [img]#transform(img)
    #     arr.append(img)
        train_arr.append(img/255)

        #for image in range(len(arr)):
         #   df = df.append({'id':id_code+str(image),'class':diagnosis}, ignore_index = True)
    #         cv2.imwrite('../working/pics_all/'+id_code+str(image)+'.png', arr[image])    
          #  train_arr.append(arr[image]/255)
df = train
    
#         diag.append(diagnosis)
#     if index%100==0:
#         plt.imshow(img,'gray')
#         print(img.shape)
#         plt.show()
    #df = df.append({'id':id_code,'x_len':img.shape[0], 'y_len': img.shape[1], 'aspect_ratio': img.shape[0]/img.shape[1], 'class':diagnosis}, ignore_index=True)
    #print(df)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


del train


# In[ ]:


shape_arr = 14648


# In[ ]:


shape_arr = df.shape[0]


# In[ ]:


work_dir = '../working/pics_all/'


# In[ ]:


# train_arr = []
# for index,value in df.iterrows():
#     id_code = value[0]
#     img = cv2.imread(work_dir+id_code+'.png',0)
#     train_arr.append(img/255)


# In[ ]:


df = df.drop('id_code',axis = 1)


# In[ ]:


df = pd.get_dummies(df.astype(str))
diag = np.array(df)
del df


# In[ ]:


train_arr = np.reshape(np.array(train_arr, dtype = 'float32'), (shape_arr,256,256,1))


# In[ ]:


# train_arr = tf.convert_to_tensor(np.reshape(np.array(train_arr),(3662,64,64,1)), dtype = 'float32')


# In[ ]:


print(len(diag))


# In[ ]:


diag = np.reshape(diag,(shape_arr,5))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,
                                            GlobalAveragePooling2D,
                                            Dense,
                                            Dropout,
                                            BatchNormalization,
                                            Conv2D,
                                            MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (CSVLogger,
                                        ModelCheckpoint,
                                        EarlyStopping)
# from tensorflow.keras.metrics import Metric
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121

from sklearn.metrics import cohen_kappa_score


# In[ ]:


def cnn_model():
    input_shape = ((256,256,1))
    inputs = layers.Input(shape=input_shape)

    base_model = ResNet50(weights=None, include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    output = Dense(5, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(optimizer = Adam(lr=0.00005), loss='categorical_crossentropy', metrics= ["accuracy"] )
    print(model.summary())
    return model


# In[ ]:


train_model = tf.keras.models.load_model("../input/aptos2019blindnessdetection-resnet50model/saved_mode500l.h5")


# In[ ]:


# gc.collect()


# In[ ]:


# train_arr = train_arr[0:100]
# diag = diag[0:100]


# In[ ]:


train_datagen = ImageDataGenerator(#rescale=1. / 255, 
                                         #validation_split=0.15, 
                                         horizontal_flip=True,
                                         vertical_flip=True, 
                                         rotation_range=20, 
                                         zoom_range=0.1, 
                                         shear_range=0.1,
                                        fill_mode='nearest')


# In[ ]:


train_generator = train_datagen.flow(x=train_arr,
                                     y = diag,
                                        batch_size=12,
                                        seed=5000
                                        )


# In[ ]:


def generator():
    while(1):
        for _i in range(0,shape_arr,32):
            yield(train_datagen.flow(train_arr[_i:_i+32],diag[_i:_i+32], batch_size=12,
                                        shuffle=True,
                                        seed=40))


# In[ ]:





# In[ ]:


test.shape


# In[ ]:


if test.shape[0]<=1928:
    train_model.fit_generator(train_generator, epochs=1, steps_per_epoch=shape_arr//10, verbose=2)
else:
    train_model.fit_generator(train_generator, epochs=100, steps_per_epoch=shape_arr//10, verbose=2)


# In[ ]:


train_model.save('saved_model_dense.h5')


# In[ ]:


del train_arr
del diag


# In[ ]:


# gc.collect()


# In[ ]:


df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


# In[ ]:


directory = '../input/aptos2019-blindness-detection/test_images/'
train_arr = []
for index,value in df.iterrows():
    id_code = value[0]
#     img = cv2.imread(directory+id_code+'.png')
    img = circle_crop_v2(directory+id_code+'.png')
#     arr = transform(img)
    arr = []
    train_arr.append(img/255)
#     train_arr.extend(arr)
        #df = df.append({'id':id_code+str(image),'class':diagnosis}, ignore_index = True)
#         cv2.imwrite('../working/pics_all/'+id_code+str(image)+'.png', arr[image])    
        
    


# In[ ]:


train_arr = np.reshape(np.array(train_arr),(len(train_arr),256,256,1))


# In[ ]:


train_arr.shape


# In[ ]:


def generator_2():
    while(1):
        for _i in range(0,len(train_arr),128):
            yield((train_arr[_i:_i+128]))


# In[ ]:


print(random.choice([0,1,2,3,4]))


# In[ ]:


diag = []


for image in train_arr:
    pred = train_model.predict(np.reshape(image,(1,256,256,1)))
    diag.append(pred)


# In[ ]:


diag = np.array(diag)


# In[ ]:


l = []
if test.shape[0]<=1928:
    l = [random.choice([0,1,2,3,4]) for i in range(df.shape[0])]
else:
    for r in diag:
            l.append(np.where(r[0]==np.max(r[0]))[0][0]) 
    


# In[ ]:


df['diagnosis'] = l


# In[ ]:


df.to_csv('submission.csv',index = False)


# In[ ]:


get_ipython().system('rm -rf pics_all')

