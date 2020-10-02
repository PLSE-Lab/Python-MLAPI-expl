#!/usr/bin/env python
# coding: utf-8

# <h1>Turkish Lira Banknote Classifier Project</h1>

# <h1>Load Datasets</h1>

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import re
import skimage 
import os
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import InceptionResNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


# In[ ]:


main_path='/kaggle/input/turkish-lira-banknote-dataset/'
train_lst, test_lst=[], []
with open ('/kaggle/input/turkish-lira-banknote-dataset/validation.txt') as test_f:
    reader=test_f.read()
    lst=reader.split()
    for img in lst:
        test_lst.append(main_path+img)
with open ('/kaggle/input/turkish-lira-banknote-dataset/train.txt') as train_f:
    reader=train_f.read()
    lst=reader.split()
    for img in lst:
        train_lst.append(main_path+img)


# In[ ]:


def show_image(path):
    comp=re.compile(r'[\w\d\/\-]+\/(\d{0,5})\/[\w\d\_\.]+')
    val=comp.findall(path)[0]
    img=plt.imread(path)
    plt.imshow(img)
    plt.title('Turkish banknote: {}'.format(val))
    plt.show()


# In[ ]:


# setting the image shape
img=plt.imread('/kaggle/input/turkish-lira-banknote-dataset/20/20_1_0005.png')
height=img.shape[0]
width=img.shape[1]
channels=img.shape[2]
print('image shape: width={}, height={}, channels={}'.format(width, width, channels))


# <p>now let's make a structured dataset and a label for each image path</p>

# In[ ]:


train_labels, test_labels=[], []
for path in train_lst:
    comp=re.compile(r'[\w\d\/\-]+\/(\d{0,5})\/[\w\d\_\.]+')
    val=comp.findall(path)[0]
    train_labels.append(val)
for path in test_lst:
    comp=re.compile(r'[\w\d\/\-]+\/(\d{0,5})\/[\w\d\_\.]+')
    val=comp.findall(path)[0]
    test_labels.append(val)
train_df=pd.DataFrame({
    'image_path':train_lst,
    'label':train_labels
})
test_df=pd.DataFrame({
    'image_path':test_lst,
    'label':test_labels
})


# In[ ]:


train_df.head()


# In[ ]:


train_df['label'].unique()


# <h1>Let's Show some images</h1>

# In[ ]:


lst=[train_df[train_df['label']=='5']['image_path'].tolist()[0],
     train_df[train_df['label']=='10']['image_path'].tolist()[0],
     train_df[train_df['label']=='20']['image_path'].tolist()[0],
     train_df[train_df['label']=='50']['image_path'].tolist()[0],
     train_df[train_df['label']=='100']['image_path'].tolist()[0],
     train_df[train_df['label']=='200']['image_path'].tolist()[0]]
for img in lst:
    show_image(img)


# <h1>Image processing</h1>

# <p>let's differentiate between the contrast of each (R, G, B) components in each banknote</p>

# In[ ]:


# banknote: 5
img_5=plt.imread(lst[0])
red_5=img_5[:, :, 0]
green_5=img_5[:,:,1]
blue_5=img_5[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_5, green_5, blue_5], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


# banknote: 10
img_10=plt.imread(lst[1])
red_10=img_10[:, :, 0]
green_10=img_10[:,:,1]
blue_10=img_10[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_10, green_10, blue_10], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


# banknote: 20
img_20=plt.imread(lst[2])
red_20=img_20[:, :, 0]
green_20=img_20[:,:,1]
blue_20=img_20[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_20, green_20, blue_20], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


# banknote: 50
img_50=plt.imread(lst[3])
red_50=img_50[:, :, 0]
green_50=img_50[:,:,1]
blue_50=img_50[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_50, green_50, blue_50], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


# banknote: 100
img_100=plt.imread(lst[4])
red_100=img_100[:, :, 0]
green_100=img_100[:,:,1]
blue_100=img_100[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_100, green_100, blue_100], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


# banknote: 200
img_200=plt.imread(lst[5])
red_200=img_200[:, :, 0]
green_200=img_200[:,:,1]
blue_200=img_200[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([red_200, green_200, blue_200], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    ax.hist(i.ravel(), label=k, color=k, bins=256)
    ax.set_title('{}'.format(k))
    plt.legend()
plt.show()    


# In[ ]:


def to_numbers(x):
    if x == '5':
        return 0
    if x == '10':
        return 1
    if x == '20':
        return 2
    if x == '50':
        return 3
    if x == '100':
        return 4
    if x == '200':
        return 5
train_df['labels']=train_df['label'].apply(to_numbers)
test_df['labels']=test_df['label'].apply(to_numbers)
train_df.drop('label', axis=1, inplace=True)
test_df.drop('label', axis=1, inplace=True)
train_ohe=to_categorical(train_df['labels'], 6)
test_ohe=to_categorical(test_df['labels'], 6)


# <h1>Modelling</h1>

# In[ ]:


K.clear_session()
model=Sequential()
model.add(Conv2D(input_shape=(180, 320 ,3),padding='same',kernel_size=3,filters=16))
model.add(LeakyReLU(0.1))
model.add(Conv2D(padding='same',kernel_size=3,filters=32))
model.add(LeakyReLU(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(padding='same',kernel_size=3,filters=32))
model.add(LeakyReLU(0.1))
model.add(Conv2D(padding='same',kernel_size=3,filters=64))
model.add(LeakyReLU(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(LeakyReLU(0.1))
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()


# In[ ]:


model.compile(
    loss='categorical_crossentropy', # this is our cross-entropy
    optimizer='adam',
    metrics=['accuracy']  # report accuracy during training
)


# In[ ]:


import cv2
def preprocess_img(df):
    image_lst=[]
    for path in df:
        img=cv2.imread(path)
        img_array = Image.fromarray(img, 'RGB')
        resized_img = img_array.resize((320, 180))
        image_lst.append(np.array(resized_img))
    return image_lst
train_imgs=preprocess_img(train_df['image_path'].tolist())
test_imgs=preprocess_img(test_df['image_path'].tolist())


# In[ ]:


train_imgs=np.array(train_imgs)
test_imgs=np.array(test_imgs)


# In[ ]:


x_train, x_test, y_train, y_test=train_test_split(train_imgs,
                                                  train_df['labels'],
                                                  test_size=0.2,
                                                  random_state=42)


# In[ ]:


y_test_ohe=to_categorical(y_test, 6)
y_train_ohe=to_categorical(y_train, 6)


# In[ ]:


model.fit(x_train,
          y_train_ohe,
          batch_size=100,
          epochs=40,
          validation_data=(x_test, y_test_ohe))


# In[ ]:


y_val_ohe=to_categorical(test_df['labels'], 6)
model.evaluate(test_imgs, y_val_ohe, batch_size=100)


# In[ ]:


pred_labels=model.predict_classes(test_imgs)


# In[ ]:


pd.DataFrame({
    'labels':test_df['labels'],
    'predicted_labels':pred_labels
})


# In[ ]:


converter={
    0:5,
    1:10,
    2:20,
    3:50,
    4:100,
    5:200
}
pred_labels_2=[]
for i in pred_labels:
    pred_labels_2.append(converter[i])


# In[ ]:


show_image(test_df['image_path'].to_list()[4])
print(pred_labels_2[4])

