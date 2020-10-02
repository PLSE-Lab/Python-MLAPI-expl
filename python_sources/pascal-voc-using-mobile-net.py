#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


from keras.preprocessing.image import img_to_array
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization
from keras.layers import MaxPooling2D,Dropout
from keras.layers import Flatten
from keras.layers import Dense
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import math
from keras.preprocessing.image import ImageDataGenerator


# In[3]:



base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(128, 128, 3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x) 
preds=Dense(20,activation='softmax')(x)


# In[4]:


model=Model(inputs=base_model.input,outputs=preds)
model.summary()


# In[5]:


for layer in model.layers[:-7]:
    layer.trainable=False
for layer in model.layers[-7:]:
    layer.trainable=True


# In[6]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


ls ../input/voctest_06-nov-2007/VOCdevkit/VOC2007/


# In[ ]:


ls ../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/


# In[ ]:



data = []
imagePaths = sorted(list(os.listdir("../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/")))
dim = (128, 128)

for img in imagePaths:
    image = cv2.imread("../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/" + img)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = img_to_array(image)
    data.append(image)

"""
imagePaths = sorted(list(os.listdir("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/")))
for img in imagePaths:
    image = cv2.imread("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/" + img)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = img_to_array(image)
    data.append(image)
"""

data = np.array(data, dtype="float32") / 255.0


# In[ ]:





# In[ ]:


data.shape


# In[ ]:





# In[ ]:


# !ls ../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/


# In[ ]:


object_list = sorted(['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                      'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle',
                      'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'])
print(len(object_list))

DF = pd.read_csv("../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", header = None, sep="\n",engine='python', dtype=str, names =['img_ID'])

DF2 = pd.read_csv("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt", header = None, sep="\n",engine='python', dtype=str, names =['img_ID'])

DF = DF.append(DF2, ignore_index=True)

for obj in object_list:

    filename = "../input/voctrainval_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/" + obj + "_trainval.txt"
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # https://www.w3schools.com/python/python_regex.asp
    df = pd.read_csv(filename, header = None, sep=r"\s*",engine='python', dtype=str, names=['img_ID', obj])
    
    filename = "../input/voctest_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/" + obj + "_test.txt"
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # https://www.w3schools.com/python/python_regex.asp
    df2 = pd.read_csv(filename, header = None, sep=r"\s*",engine='python', dtype=str, names=['img_ID', obj])
    
    df = df.append(df2, ignore_index=True)
    df[obj] = df[obj].astype(int)
    df[obj][df[obj]<0] = 0
    # df.dtypes
    # df.head()
    DF = pd.merge(DF, df, on='img_ID')

DF.head(10)


# In[ ]:





# In[ ]:



labels = []
for i in range(len(DF)):
    labels.append(list(DF.iloc[i][1:]))

labels = np.array(labels)

l = labels[0:5011]


# In[ ]:


l.shape


# In[ ]:


data.shape


# In[ ]:





# In[ ]:





# In[ ]:


plt.imshow(data[0])
plt.show()


# In[ ]:



model.fit(data, l, epochs=15, validation_split=0.1, batch_size=50)


# In[ ]:



test_data = []
imagePaths = sorted(list(os.listdir("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/")))
dim = (128,128)

for img in imagePaths:
    image = cv2.imread("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/" + img)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = img_to_array(image)
    test_data.append(image)
    

test_data = np.array(test_data, dtype="float32") / 255.0
test_y = labels[5011:]


# In[ ]:



pred_l = model.predict(test_data, batch_size=None, verbose=0)

tol = 0.5
pred_label = (pred_l > tol) * 1

print(pred_label[1])
print(test_y[1])


# In[ ]:



total = 0
accurate = 0
accurate_index = []
wrong_index = []

for i in range(len(pred_label)):
    if (pred_label[i] == test_y[i]).all():
        accurate += 1
        accurate_index.append(i)
    else:
        wrong_index.append(i)
        
    total += 1
    
    
print('Total test data;', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')


# # cosine similarity

# In[ ]:



import math

total = 0
cosine_num = 0
cosine_den = 0
cosine = 0

for i in range(len(pred_label)):
    
    cosine_num = np.vdot(pred_label[i], test_y[i])
    cosine_den = math.sqrt(sum(pred_label[i])) * math.sqrt(sum(test_y[i]))
    
    if cosine_den:
        cosine = cosine + (cosine_num/ cosine_den)
    
    
    
    total += 1
    
print('Avg Cosine Similarity:', round(cosine/total, 3))


# In[ ]:





# # Some Samples

# In[ ]:



object_list = sorted(['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                      'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle',
                      'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'])

im_idx = random.sample(accurate_index, k=9)
object_np_array = np.array(object_list)

nrows = 3
ncols = 3
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))

n = 0
for row in range(nrows):
    for col in range(ncols):
            ax[row,col].imshow(test_data[im_idx[n]])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(tuple(object_np_array[pred_label[im_idx[n]]==1]), tuple(object_np_array[test_y[im_idx[n]]==1])))
            n += 1

plt.show()


# In[ ]:




im_idx = random.sample(wrong_index, k=9)

nrows = 3
ncols = 3
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))

n = 0
for row in range(nrows):
    for col in range(ncols):
            ax[row,col].imshow(test_data[im_idx[n]])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(tuple(object_np_array[pred_label[im_idx[n]]==1]), tuple(object_np_array[test_y[im_idx[n]]==1])))
            n += 1

plt.show()


# In[ ]:





# In[ ]:



(x_train,x_test,y_train,y_test)=train_test_split(data,l,test_size=0.1,shuffle=False)
print(x_train.shape, x_test.shape)


# # augmentation

# In[ ]:


train_datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

batch_size = 50
model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size),
                    steps_per_epoch=math.ceil(len(x_train)/batch_size), epochs = 15, validation_data=(x_test, y_test))


# In[ ]:


model.save('CNN_pascal_imagegenerator_5epoch.h5')


# In[ ]:


pred_l_gn = model.predict(test_data)

tol = 0.5
pred_label_gn = (pred_l_gn > tol) * 1

print(pred_label_gn[1])
print(test_y[1])

total = 0
accurate = 0
accurate_index = []
wrong_index = []

for i in range(len(pred_label)):
    if (pred_label_gn[i] == test_y[i]).all():
        accurate += 1
        accurate_index.append(i)
    else:
        wrong_index.append(i)
        
    total += 1
    
    
print('Total test data;', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')


# In[ ]:


import math

total = 0
cosine_num = 0
cosine_den = 0
cosine = 0

for i in range(len(pred_label_gn)):
    
    cosine_num = np.vdot(pred_label_gn[i], test_y[i])
    cosine_den = math.sqrt(sum(pred_label_gn[i])) * math.sqrt(sum(test_y[i]))
    
    if cosine_den:
        cosine = cosine + (cosine_num/ cosine_den)
    
    
    
    total += 1
    
print('Avg Cosine Similarity:', round(cosine/total, 3))


# In[ ]:


object_list = sorted(['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                      'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle',
                      'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'])

im_idx = random.sample(wrong_index, k=9)
object_np_array = np.array(object_list)

nrows = 3
ncols = 3
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))

n = 0
for row in range(nrows):
    for col in range(ncols):
            ax[row,col].imshow(test_data[im_idx[n]])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(tuple(object_np_array[pred_label_gn[im_idx[n]]==1]), tuple(object_np_array[test_y[im_idx[n]]==1])))
            n += 1

plt.show()

