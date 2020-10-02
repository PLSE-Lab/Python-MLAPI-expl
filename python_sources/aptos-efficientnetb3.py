#!/usr/bin/env python
# coding: utf-8

# <h1>APTOS Diabetic Retinopathy Severity Prediction</h1>
# In this kernel we will EfficientNetB3 which I pretrained on 2015 dataset and fine-tune on 2019 dataset for 30 epochs.
# <br></br>
# We will also perform pseudo labeling (which I wish I had tried during competition, I still have a lot to learn though :D)

# In[ ]:


'''
    Build and install EfficientNet
    Import necessary libraries
'''
get_ipython().system('pip install ../input/efficientnet/efficientnet-master/efficientnet-master')
from efficientnet.keras import *

import cv2
import numpy as np 

import pandas as pd
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.utils import *
from keras.layers import *

from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc


# In[ ]:


'''
    Config
'''
IMG_SIZE = 300
BATCH_SIZE = 16


# In[ ]:


'''
    Preprocessing using Ben Graham's method (Last competition's winner) 
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
'''
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    
def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image).astype('uint8')
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image.astype('float32') / 255.

'''
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
'''
def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype('uint8')
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image.astype('float32') / 255.


# In[ ]:


'''
    Initialize EfficientNetB3
    Output relu is relu rectifier with max_value of 4 to restrain predictions to proper range
'''
def output_relu(x):
    return K.relu(x, max_value=4)

base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation=output_relu, kernel_initializer='he_normal')(x)
model = Model(inputs=base_model.input, outputs=x)


# In[ ]:


'''
    Load pretrained weights
'''
model.load_weights('../input/pretrained-weights/model_b3_epochs_30_300.h5')


# In[ ]:


'''
    Load training df and test df for pseudo labeling
'''
train_csv = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_id_codes = train_csv['id_code']
train_labels = train_csv['diagnosis']

for i in range(len(train_id_codes)):
    train_id_codes[i] = '../input/aptos2019-blindness-detection/train_images/{}.png'.format(train_id_codes[i])

test_csv = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
test_id_codes = test_csv['id_code']
test_pseudo_labels = np.empty(len(test_id_codes), dtype='float32')
for i in range(len(test_id_codes)):
    test_id_codes[i] = '../input/aptos2019-blindness-detection/test_images/{}.png'.format(test_id_codes[i])
    img = cv2.imread(test_id_codes[i])
    img = load_ben_color(img)
    X = np.array([img])
    pred = model.predict(X)
    test_pseudo_labels[i] = pred


# In[ ]:


'''
    Float to integer prediction function. In this case round to nearest integer
'''
def predict(X, coef=[0.5, 1.5, 2.5, 3.5]):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p


# In[ ]:


'''
    Round pseudo_labels
'''
test_pseudo_labels = predict(test_pseudo_labels).astype('uint8')
print(test_pseudo_labels)


# In[ ]:


'''
    Create DataFrame for ImageDataGenerator
    Combine training and test images with pseudo labels
'''
d = {}
d['id_code'] = np.concatenate((train_id_codes, test_id_codes), axis=0)
d['diagnosis'] = np.concatenate((train_labels, test_pseudo_labels), axis=0).astype('str')
df = pd.DataFrame(data=d)


# In[ ]:


'''
    Create Image Data Generator
'''
gen = ImageDataGenerator(preprocessing_function=preprocessing)
pseudo_datagen = gen.flow_from_dataframe(df, directory='.', x_col='id_code', y_col='diagnosis', target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=16)


# In[ ]:


'''
    Double check preprocessing function
'''
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))
it = 0
for x, y in pseudo_datagen:
    ax[it].imshow((x[0]*255.).astype('uint8'))
    ax[it].axis('off')
    it += 1
    if it == 4:
        break


# In[ ]:


'''
    Compiling model and retraining for 5 epochs
'''
model.compile(optimizer=keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True, decay=1e-6), loss='mse')
model.fit_generator(pseudo_datagen,
                    steps_per_epoch=len(pseudo_datagen),
                    epochs=5)


# In[ ]:


'''
    Final prediction on testset
'''
test_csv = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
id_code = test_csv['id_code']
test_prediction = np.empty(len(id_code), dtype='float32')
for i in range(len(id_code)):
    img = cv2.imread('../input/aptos2019-blindness-detection/test_images/{}.png'.format(id_code[i]))
    img = load_ben_color(img)
    X = np.array([img])
    pred = model.predict(X)
    test_prediction[i] = pred


# In[ ]:


'''
    Submission
'''
prediction = predict(test_prediction).astype('uint8')
test_csv['diagnosis'] = prediction
test_csv.to_csv("submission.csv", index=False)
unique, counts = np.unique(prediction, return_counts=True)
tmp = dict(zip(unique, counts))
print(tmp)
print('Done!')


# In[ ]:





# In[ ]:




