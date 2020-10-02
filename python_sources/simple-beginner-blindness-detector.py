#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sample=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot(x='diagnosis',data=train)


# 
# 
#     0 - No DR
# 
#     1 - Mild
# 
#     2 - Moderate
# 
#     3 - Severe
# 
#     4 - Proliferative DR
# 

# #### Image Handling
# 

# In[ ]:


import cv2
import glob

X_data = []
images = glob.glob ("../input/train_images/*.png")


# In[ ]:


images[0:5]


# In[ ]:


import random
r = random.sample(images, 3)
r

plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.imread(r[0]))

plt.subplot(132)
plt.imshow(cv2.imread(r[1]))

plt.subplot(133)
plt.imshow(cv2.imread(r[2]))


# ##### wow our eyes are really beautiful :P

# ### I'll update this kernel soon :)

# #### Image path

# In[ ]:


train_path = '../input/train_images/'
test_path = '../input/test_images/'


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import os
import cv2


# In[ ]:


train.head()


# In[ ]:


id_code=train['id_code'].values
diagnosis=train['diagnosis'].values


train=[]
X=[]
Y=[]
a=0
IMG_SIZE=150
for i in tqdm(sorted(os.listdir(train_path))):
    path=os.path.join(train_path,i)
    i=cv2.imread(path,cv2.IMREAD_COLOR)
    i = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
    X.append(i)
    train.append([np.array(diagnosis),diagnosis[a]])
    a=a+1

train=np.array(train)
Y=train[:,1]
train=train[:,0]
X=np.array(X)

X.shape

X=X/255
train=train/255


# In[ ]:




test1=[]
X_test=[]
IMG_SIZE=150
for i in tqdm(os.listdir(test_path)):
    id_code=i
    path=os.path.join(test_path,i)
    i=cv2.imread(path,cv2.IMREAD_COLOR)
    i = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
    X_test.append(i)
    test1.append([np.array(i),id_code])

X_test=np.array(X_test)
X_test.shape
test1=np.array(test1)
id_test=test1[:,1]
test1=test1[:,0]
test1.shape

X_test=X_test/255
test1=test1/255


# In[ ]:


model = Sequential()
model.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu",input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=2,strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2,strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2,strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(1,activation="softmax"))
model.summary()


# In[ ]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
h=model.fit(X,Y,batch_size=256,validation_split=0.2,epochs=100)


# In[ ]:




