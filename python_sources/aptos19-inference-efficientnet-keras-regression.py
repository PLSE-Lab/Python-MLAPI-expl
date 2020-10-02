#!/usr/bin/env python
# coding: utf-8

# # Inference EfficientNet Keras - Regression
# 
# ---
# 
# This is an inference kernel. You can find the training one **[HERE](https://www.kaggle.com/raimonds1993/aptos19-efficientnet-keras-regression)**.
# 
# ### If you enjoyed the kernel, <span style="color:red">please upvote :)</span>.
# 
# ### Credits
# 
# - [Efficient Net weights](https://www.kaggle.com/ratthachat/efficientnet-keras-weights-b0b5), by **Neuron Engineer**.

# In[ ]:


import os
import sys
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
print(os.listdir('../input'))


im_size = 224

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(test_df.shape)


# In[ ]:


# utility functions
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def preprocess_image(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)
    
    return img


# # Test Data

# In[ ]:


N = test_df.shape[0]
x_test = np.empty((N, im_size, im_size, 3), dtype=np.uint8)

try:
    for i, image_id in enumerate(test_df['id_code']):
        x_test[i, :, :, :] = preprocess_image(
            f'../input/aptos2019-blindness-detection/test_images/{image_id}.png',
            desired_size=im_size
        )
    print('Test dataset correctly processed')
except:
    print('Test dataset NOT processed')


# # Model: EffNetB5

# In[ ]:


print(os.listdir("../input/kerasefficientnetsmaster/keras-efficientnets-master/keras-efficientnets-master/keras_efficientnets"))
sys.path.append(os.path.abspath('../input/kerasefficientnetsmaster/keras-efficientnets-master/keras-efficientnets-master/'))
from keras_efficientnets import EfficientNetB5
effnet = EfficientNetB5(input_shape=(im_size,im_size,3),
                        weights=sys.path.append(os.path.abspath('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')),
                        include_top=False)


def build_model():
    model = Sequential()
    model.add(effnet)
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='linear'))
    return model

model = build_model()

model.load_weights('../input/efficientnet-keras-aptos/model.h5')
model.summary()


# # Submission

# In[ ]:


y_test = model.predict(x_test)


coef = [0.5, 1.5, 2.5, 3.5]

# Optimized on validation set
#coef = [0.5370942, 1.51580731, 2.61728832, 3.37044039]

for i, pred in enumerate(y_test):
    if pred < coef[0]:
        y_test[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        y_test[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        y_test[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        y_test[i] = 3
    else:
        y_test[i] = 4

test_df['diagnosis'] = y_test.astype(int)
test_df.to_csv('submission.csv',index=False)

print(round(test_df.diagnosis.value_counts()/len(test_df)*100,4))

