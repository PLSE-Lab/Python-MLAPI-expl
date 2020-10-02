#!/usr/bin/env python
# coding: utf-8

# In[1]:




import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D

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

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications import MobileNet



# In[2]:



base_model=InceptionResNetV2(weights='imagenet',include_top=False, input_shape=(128, 128, 3), pooling='avg')
# for layer in base_model.layers:
#     layer.trainable = False
for layer in base_model.layers[:-3]:
    layer.trainable=False
for layer in base_model.layers[-3:]:
    layer.trainable=True
base_model.summary()


# In[3]:



x = base_model.output
# x=GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)

dd = 0.5

# different model heads
# https://github.com/sugi-chan/fgo-multi-task-keras/blob/master/fgo_multiclass.ipynb
y1 = Dense(128, activation='relu')(x)
y1 = Dropout(dd)(y1)
y1 = Dense(64, activation='relu')(y1)
y1 = Dropout(dd)(y1)

y2 = Dense(128, activation='relu')(x)
y2 = Dropout(dd)(y2)
y2 = Dense(64, activation='relu')(y2)
y2 = Dropout(dd)(y2)

y3 = Dense(128, activation='relu')(x)
y3 = Dropout(dd)(y3)
y3 = Dense(64, activation='relu')(y3)
y3 = Dropout(dd)(y3)

y4 = Dense(128, activation='relu')(x)
y4 = Dropout(dd)(y4)
y4 = Dense(64, activation='relu')(y4)
y4 = Dropout(dd)(y4)

y5 = Dense(128, activation='relu')(x)
y5 = Dropout(dd)(y5)
y5 = Dense(64, activation='relu')(y5)
y5 = Dropout(dd)(y5)

y6 = Dense(128, activation='relu')(x)
y6 = Dropout(dd)(y6)
y6 = Dense(64, activation='relu')(y6)
y6 = Dropout(dd)(y6)

y7 = Dense(128, activation='relu')(x)
y7 = Dropout(dd)(y7)
y7 = Dense(64, activation='relu')(y7)
y7 = Dropout(dd)(y7)

y8 = Dense(128, activation='relu')(x)
y8 = Dropout(dd)(y8)
y8 = Dense(64, activation='relu')(y8)
y8 = Dropout(dd)(y8)

y9 = Dense(128, activation='relu')(x)
y9 = Dropout(dd)(y9)
y9 = Dense(64, activation='relu')(y9)
y9 = Dropout(dd)(y9)

y10 = Dense(128, activation='relu')(x)
y10 = Dropout(dd)(y10)
y10 = Dense(64, activation='relu')(y10)
y10 = Dropout(dd)(y10)

y11 = Dense(128, activation='relu')(x)
y11 = Dropout(dd)(y11)
y11 = Dense(64, activation='relu')(y11)
y11 = Dropout(dd)(y11)

y12 = Dense(128, activation='relu')(x)
y12 = Dropout(dd)(y12)
y12 = Dense(64, activation='relu')(y12)
y12 = Dropout(dd)(y12)

y13 = Dense(128, activation='relu')(x)
y13 = Dropout(dd)(y13)
y13 = Dense(64, activation='relu')(y13)
y13 = Dropout(dd)(y13)

y14 = Dense(128, activation='relu')(x)
y14 = Dropout(dd)(y14)
y14 = Dense(64, activation='relu')(y14)
y14 = Dropout(dd)(y14)

y15 = Dense(128, activation='relu')(x)
y15 = Dropout(dd)(y15)
y15 = Dense(64, activation='relu')(y15)
y15 = Dropout(dd)(y15)

y16 = Dense(128, activation='relu')(x)
y16 = Dropout(dd)(y16)
y16 = Dense(64, activation='relu')(y16)
y16 = Dropout(dd)(y16)

y17 = Dense(128, activation='relu')(x)
y17 = Dropout(dd)(y17)
y17 = Dense(64, activation='relu')(y17)
y17 = Dropout(dd)(y17)

y18 = Dense(128, activation='relu')(x)
y18 = Dropout(dd)(y18)
y18 = Dense(64, activation='relu')(y18)
y18 = Dropout(dd)(y18)

y19 = Dense(128, activation='relu')(x)
y19 = Dropout(dd)(y19)
y19 = Dense(64, activation='relu')(y19)
y19 = Dropout(dd)(y19)

y20 = Dense(128, activation='relu')(x)
y20 = Dropout(dd)(y20)
y20 = Dense(64, activation='relu')(y20)
y20 = Dropout(dd)(y20)




#connect all the heads to their final output layers
y1 = Dense(1, activation='sigmoid',name= 'aeroplane')(y1)
y2 = Dense(1, activation='sigmoid',name= 'bicycle')(y2)
y3 = Dense(1, activation='sigmoid',name= 'bird')(y3)
y4 = Dense(1, activation='sigmoid',name= 'boat')(y4)
y5 = Dense(1, activation='sigmoid',name= 'bottle')(y5)
y6 = Dense(1, activation='sigmoid',name= 'bus')(y6)
y7 = Dense(1, activation='sigmoid',name= 'car')(y7)
y8 = Dense(1, activation='sigmoid',name= 'cat')(y8)
y9 = Dense(1, activation='sigmoid',name= 'chair')(y9)
y10 = Dense(1, activation='sigmoid',name= 'cow')(y10)
y11 = Dense(1, activation='sigmoid',name= 'diningtable')(y11)
y12 = Dense(1, activation='sigmoid',name= 'dog')(y12)
y13 = Dense(1, activation='sigmoid',name= 'horse')(y13)
y14 = Dense(1, activation='sigmoid',name= 'motorbike')(y14)
y15 = Dense(1, activation='sigmoid',name= 'person')(y15)
y16 = Dense(1, activation='sigmoid',name= 'pottedplant')(y16)
y17 = Dense(1, activation='sigmoid',name= 'sheep')(y17)
y18 = Dense(1, activation='sigmoid',name= 'sofa')(y18)
y19 = Dense(1, activation='sigmoid',name= 'train')(y19)
y20 = Dense(1, activation='sigmoid',name= 'tvmonitor')(y20)


# ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor']

# In[4]:



model = Model(inputs=base_model.input, outputs=[y1, y2, y3, y4, y5,y6,y7,y8,y9,y10,y11, y12, y13, y14, y15,y16,y17,y18,y19,y20])

# model.summary()


# In[ ]:





# In[5]:




model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
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


data = np.array(data, dtype="float32") / 255.0


# In[ ]:


print(data.shape)


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


DF.describe()


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


a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,a16,a17,a18,a19,a20 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []



for i in range(len(DF)):
    a1.append((DF.iloc[i]['aeroplane']))
    a2.append((DF.iloc[i]['bicycle']))
    a3.append((DF.iloc[i]['bird']))
    a4.append((DF.iloc[i]['boat']))
    a5.append((DF.iloc[i]['bottle']))
    a6.append((DF.iloc[i]['bus']))
    a7.append((DF.iloc[i]['car']))
    a8.append((DF.iloc[i]['cat']))
    a9.append((DF.iloc[i]['chair']))
    a10.append((DF.iloc[i]['cow']))
    a11.append((DF.iloc[i]['diningtable']))
    a12.append((DF.iloc[i]['dog']))
    a13.append((DF.iloc[i]['horse']))
    a14.append((DF.iloc[i]['motorbike']))
    a15.append((DF.iloc[i]['person']))
    a16.append((DF.iloc[i]['pottedplant']))
    a17.append((DF.iloc[i]['sheep']))
    a18.append((DF.iloc[i]['sofa']))
    a19.append((DF.iloc[i]['train']))
    a20.append((DF.iloc[i]['tvmonitor']))
    

a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)
a4 = np.array(a4)
a5 = np.array(a5)
a6 = np.array(a6)
a7 = np.array(a7)
a8 = np.array(a8)
a9 = np.array(a9)
a10 = np.array(a10)
a11 = np.array(a11)
a12 = np.array(a12)
a13 = np.array(a13)
a14 = np.array(a14)
a15 = np.array(a15)
a16 = np.array(a16)
a17 = np.array(a17)
a18 = np.array(a18)
a19 = np.array(a19)
a20 = np.array(a20)

a1_t = a1[0:5011]
a2_t = a2[0:5011]
a3_t = a3[0:5011]
a4_t = a4[0:5011]
a5_t = a5[0:5011]
a6_t = a6[0:5011]
a7_t = a7[0:5011]
a8_t = a8[0:5011]
a9_t = a9[0:5011]
a10_t = a10[0:5011]
a11_t = a11[0:5011]
a12_t = a12[0:5011]
a13_t = a13[0:5011]
a14_t = a14[0:5011]
a15_t = a15[0:5011]
a16_t = a16[0:5011]
a17_t = a17[0:5011]
a18_t = a18[0:5011]
a19_t = a19[0:5011]
a20_t = a20[0:5011]





# In[ ]:





# In[ ]:


plt.imshow(data[0])
plt.show()


# In[ ]:



model.fit(data, [a1_t, a2_t, a3_t, a4_t, a5_t, a6_t, a7_t, a8_t, a9_t, a10_t, a11_t, a12_t, a13_t, a14_t, a15_t, a16_t, a17_t, a18_t, a19_t, a20_t],
          verbose=2, epochs=20, validation_split=0.1, batch_size=50)


# In[ ]:





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


# In[ ]:



labels = []
for i in range(len(DF)):
    labels.append(list(DF.iloc[i][1:]))

labels = np.array(labels)

test_y = labels[5011:]


# In[ ]:



a1_p, a2_p, a3_p, a4_p, a5_p, a6_p, a7_p, a8_p, a9_p, a10_p, a11_p, a12_p, a13_p, a14_p, a15_p, a16_p, a17_p, a18_p, a19_p, a20_p = model.predict(test_data, batch_size=None, verbose=0)

tol = 0.5
print(a1_p[1])
print(test_y[1][0])


# In[ ]:





# In[ ]:



a1_p = (a1_p > tol) * 1
a2_p = (a2_p > tol) * 1
a3_p = (a3_p > tol) * 1
a4_p = (a4_p > tol) * 1
a5_p = (a5_p > tol) * 1
a6_p = (a6_p > tol) * 1
a7_p = (a7_p > tol) * 1
a8_p = (a8_p > tol) * 1
a9_p = (a9_p > tol) * 1
a10_p = (a10_p > tol) * 1
a11_p = (a11_p > tol) * 1
a12_p = (a12_p > tol) * 1
a13_p = (a13_p > tol) * 1
a14_p = (a14_p > tol) * 1
a15_p = (a15_p > tol) * 1
a16_p = (a16_p > tol) * 1
a17_p = (a17_p > tol) * 1
a18_p = (a18_p > tol) * 1
a19_p = (a19_p > tol) * 1
a20_p = (a20_p > tol) * 1

pred_label = []
for i in range(len(a1_p)):
    pred_label.append([int(a1_p[i]), int(a2_p[i]), int(a3_p[i]), int(a4_p[i]), int(a5_p[i]), int(a6_p[i]), int(a7_p[i]), int(a8_p[i]), int(a9_p[i]), int(a10_p[i]),
                      int(a11_p[i]), int(a12_p[i]), int(a13_p[i]), int(a14_p[i]), int(a15_p[i]), int(a16_p[i]), int(a17_p[i]), int(a18_p[i]), int(a19_p[i]), int(a20_p[i])])
    
pred_label = np.array(pred_label)
print(len(pred_label))


# In[ ]:


# object_np_array[pred_label[3]==1.0]
print(pred_label[3].dtype)
print(pred_label[3] == 1.0)


# In[ ]:


print(pred_label[3])
print(test_y[3])
print(test_y[3] == pred_label[3])
plt.imshow(test_data[3])


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



object_np_array = np.array(object_list)

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




im_idx = random.sample(accurate_index, k=9)

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





# <h3><center> THE END

# In[ ]:




