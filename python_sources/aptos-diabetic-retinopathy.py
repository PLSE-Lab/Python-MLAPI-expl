#!/usr/bin/env python
# coding: utf-8

# <h1>Diabetic Retinopathy</h1>
# <p>Diabetic retinopathy is damage to the retina caused by complications of diabetes mellitus.
# The condition can lead to blindness if left untreated. Early blindness due to diabetic retinopathy (DR) is usually preventable with routine checks and effective management of the underlying diabetes.</p>
# <br><br><p>The retina is the membrane that covers the back of the eye. It is highly sensitive to light.
# <br>
# It converts any light that hits the eye into signals that can be interpreted by the brain. This process produces visual images, and it is how sight functions in the human eye.
# <br>
# Diabetic retinopathy damages the blood vessels within the retinal tissue, causing them to leak fluid and distort vision.</p>
# 
# <h4>Data description from the competition:</h4>
# 
# <p>You are provided with a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. >Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).
# 
# A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
# <ol><li>
#     No DR</li>
# <li>
# Mild
#     </li>
#     <li>
# Moderate
#     </li>
# <li>Severe
#     </li>
#     <li>Proliferative DR</li>
# </ol>
# Your task is to create an automated analysis system capable of assigning a score based on this scale.
# 
# </p>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import glob
import cv2
from keras.utils import to_categorical
import keras


# In[ ]:


os.listdir('../input/')


# In[ ]:


train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')


# In[ ]:


train.head()


# In[ ]:


len(train)


# In[ ]:


train_list = [[Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png'),j] for i,j in zip(train.id_code[:5],train.diagnosis[:5])]
train_list


# In[ ]:


for i,j in train_list:
    plt.figure(figsize=(5,3))
    i = cv2.resize(np.asarray(i),(256,256))
    plt.title(j)
    plt.imshow(i)
    plt.show


# In[ ]:


x_train = [cv2.resize(np.asarray(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png')),(256,256)) for i in train.id_code]


# In[ ]:


x_train = np.array(x_train)


# In[ ]:


y_train = train.diagnosis


# In[ ]:


y_train = to_categorical(y_train)
y_train


# In[ ]:


model = keras.applications.densenet.DenseNet121(input_shape=(256,256,3),include_top=True,weights=None)


# In[ ]:


model.summary()


# In[ ]:


model.load_weights('../input/densenet-keras/DenseNet-BC-121-32.h5')


# In[ ]:


x = model.layers[-2].output
d = keras.layers.Dense(512,activation='relu')(x)
e = keras.layers.Dense(5,activation='softmax')(d)


# In[ ]:


model1 = keras.models.Model(model.input,e)


# In[ ]:


model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model1.fit(x_train,y_train,validation_split=0.20,epochs=10)


# In[ ]:


test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test = []
for i in test_df.id_code:
    temp = np.array(cv2.resize(np.array(Image.open('../input/aptos2019-blindness-detection/test_images/'+i+'.png')),(256,256)))
    test.append(temp)
test = np.array(test)


# In[ ]:


np.random.seed(42)
result = model1.predict(test)


# In[ ]:


res = []
for i in result:
    res.append(np.argmax(i))


# In[ ]:


df_test = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": res})
df_test.head(20)


# In[ ]:


df_test.to_csv('submission.csv',index=False)

