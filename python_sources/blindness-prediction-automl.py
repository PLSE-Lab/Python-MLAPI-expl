#!/usr/bin/env python
# coding: utf-8

# This is an example of **AutoML** usage and how **Autokeras** can be adapted for an internet disabled competition. Autokeras model is trained externally with multiple GPUs and 12 hours time. Overall, **44 different models** are tested to find the best model. Both pre-trained model weights and autokeras framework are added as external data in this kernel. 
# 
# Trained model got **65.58%** accuracy on train set, **63.45%** on validation set. 
# 
# However, this competition evaluates your kernel with a quadratic weighted kappa score. It got **0.4264** cohen kappa score on its own validation data. Besides, it got **0.114** kappa score on public test set, **0.385** kappa score on private test set. 
# 
# It seems that autokeras is **promising** for this kind of competitions but you should let it to be trained for a much longer time.
# 
# If you are interested in AutoML and Autokeras, you should read the following blog posts.
# 
# [1] https://sefiks.com/2019/04/08/a-gentle-introduction-to-auto-keras/
# 
# [2] https://sefiks.com/2019/09/03/tips-for-building-automl-with-autokeras/

# In[ ]:


import sys
package_dir = '../input/autokeras/autokeras'
sys.path.insert(0, package_dir)


# In[ ]:


import autokeras as ak
from autokeras.utils import pickle_from_file


# In[ ]:


autokeras_model = pickle_from_file("../input/autokeras-model-v3/autokeras_model_v3.h5")


# In[ ]:


"""
# you can also convert the model to Keras but pre-trained weights disappear in this case
autokeras_model.export_keras_model('keras_model.h5')

from keras.models import load_model
keras_model = load_model('keras_model.h5')
keras_model.summary()
"""


# In[ ]:


import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# In[ ]:


IMG_SIZE = 224


# # Test set

# In[ ]:


df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")


# In[ ]:


df.head()


# In[ ]:


#https://www.kaggle.com/taindow/pre-processing-train-and-test-images

def crop_image_from_gray(img,tol=7):
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

def circle_crop_v2(img):
    img = cv2.imread(img)
    img = crop_image_from_gray(img)

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

    return img

def getImagePixelsNew(img_name):
    
    image = circle_crop_v2('../input/aptos2019-blindness-detection/test_images/%s.png' % img_name)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    #normalize in scale of 0, +1
    img_pixels = image / 255 #0 to 254
    
    return img_pixels


# In[ ]:


df['pixels'] = df['id_code'].apply(getImagePixelsNew)


# In[ ]:


#df.head()


# In[ ]:


df.iloc[0]['pixels'].shape


# In[ ]:


features = []

pbar = tqdm(range(0, df.shape[0]), desc='Processing')
for index in pbar:
    features.append(df.iloc[index]['pixels'])

print("features variable created: ",len(features))


# In[ ]:


predictions = []

pbar = tqdm(range(0, len(features)), desc='Processing')

for index in pbar:
    prediction = autokeras_model.predict(np.expand_dims(features[index], axis = 0))
    predictions.append(prediction[0])

#predictions = autokeras_model.predict(features)


# In[ ]:


predictions[0:10]


# In[ ]:


df['diagnosis'] = predictions
df = df.drop(columns=['pixels'])


# In[ ]:


df.head()


# In[ ]:


df.to_csv("submission.csv", index=False)

