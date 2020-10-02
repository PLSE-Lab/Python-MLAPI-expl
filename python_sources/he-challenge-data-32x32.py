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


train=pd.read_csv("../input/flower-recognition-he/he_challenge_data/data/train.csv")
test=pd.read_csv("../input/flower-recognition-he/he_challenge_data/data/test.csv")


# In[ ]:


test.head(5)


# In[ ]:


import numpy as np
import cv2
img = cv2.imread('../input/flower-recognition-he/he_challenge_data/data/train/0.jpg')
import matplotlib.pyplot as plt
print(img.shape)
plt.imshow(img)


# In[ ]:


from PIL import Image 
def rewind1(image_path,dataset,desired_size=32):
    img=cv2.imread(f"../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
    img=cv2.resize(img,(32,)*2).astype('uint8')
   # img=Image.open("../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
   # img=img.resize((32,32))
    return img


# In[ ]:


from PIL import Image 
def rewind2(image_path,dataset,desired_size=64):
    img=cv2.imread(f"../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
    img=cv2.resize(img,(64,)*2).astype('uint8')
   # img=Image.open("../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
   # img=img.resize((32,32))
    return img


# In[ ]:


from PIL import Image 
def rewind3(image_path,dataset,desired_size=64):
    img=cv2.imread(f"../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
    img=cv2.resize(img,(128,)*2).astype('uint8')
   # img=Image.open("../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
   # img=img.resize((32,32))
    return img


# In[ ]:


from PIL import Image 
def rewind4(image_path,dataset,desired_size=64):
    img=cv2.imread(f"../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
    img=cv2.resize(img,(256,)*2).astype('uint8')
   # img=Image.open("../input/flower-recognition-he/he_challenge_data/data/{dataset}/{image_path}.jpg")
   # img=img.resize((32,32))
    return img


# In[ ]:


# train_resized_images_32=[]
# test_resized_images_32=[]
# for img_id in train["image_id"]:
#     train_resized_images_32.append(rewind1(img_id,'train'))
# for img_id in test["image_id"]:
#     test_resized_images_32.append(rewind1(img_id,'test')) 


# In[ ]:


# train_resized_images_64=[]
# test_resized_images_64=[]
# for img_id in train["image_id"]:
#     train_resized_images_64.append(rewind2(img_id,'train'))
# for img_id in test["image_id"]:
#     test_resized_images_64.append(rewind2(img_id,'test')) 


# In[ ]:


# train_resized_images_128=[]
# test_resized_images_128=[]
# for img_id in train["image_id"]:
#     train_resized_images_128.append(rewind3(img_id,'train'))
# for img_id in test["image_id"]:
#     test_resized_images_128.append(rewind3(img_id,'test'))


# In[ ]:


train_resized_images_256=[]
test_resized_images_256=[]
for img_id in train["image_id"]:
    train_resized_images_256.append(rewind4(img_id,'train'))
for img_id in test["image_id"]:
    test_resized_images_256.append(rewind4(img_id,'test'))


# In[ ]:


# X_train_32=np.stack(train_resized_images_32)
# X_test_32=np.stack(test_resized_images_32)
# X_train_64=np.stack(train_resized_images_64)
# X_test_64=np.stack(test_resized_images_64)
# X_train_128=np.stack(train_resized_images_128)
# X_test_128=np.stack(test_resized_images_128)
X_train_256=np.stack(train_resized_images_256)
X_test_256=np.stack(test_resized_images_256)
y=train['category']
Y_train=pd.get_dummies(y,columns=[0])


# In[ ]:


print(X_test_256.shape,"     ",X_train_256.shape,"     ",Y_train.shape,"      ")


# In[ ]:


# np.save("X_train_32.npy",X_train_32)
# np.save("X_test_32.npy",X_test_32)
# np.save("X_test_64.npy",X_test_64)
# np.save("X_train_64.npy",X_train_64)
# np.save("X_test_128.npy",X_test_128)
# np.save("X_train_128.npy",X_train_128)
np.save("X_test_256.npy",X_test_256)
np.save("X_train_256.npy",X_train_256)
Y_train.to_csv('Y_train.csv')
np.save("Y_train_np.npy",Y_train)


# In[ ]:


k=1000
plt.imshow(X_train_256[k])
print(train['category'][k])


# In[ ]:


yy=train['category']
yy=pd.get_dummies(yy,columns=['category'])


# In[ ]:


# yy


# In[ ]:


kk=yy.idxmax( axis=1, skipna=True)
# kk


# In[ ]:


sales = [{1},{2},{3},{10}]
df = pd.DataFrame(sales)
df = pd.get_dummies(df, columns=[0])
# y=df['Mar'].argmax(axis=1)
kk=df.idxmax( axis=1, skipna=True)
print(kk)

