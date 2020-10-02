#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 80000 # the number of images we use from each of the two classes
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_data = pd.read_csv('../input/pices-de-monnaie/train.csv')
print(df_data.shape)


# In[ ]:


label_counts=df_data['label'].value_counts()
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[ ]:


df_data.head()


# In[ ]:


y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


# In[ ]:


df_train['label'].value_counts()


# In[ ]:


no_tumor_tissue = os.path.join(train_dir, 'dinar')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'deuxdinars')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'cinqdinars')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'dixmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'vingtmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'cinqentesmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'centmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'deuxcentsmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'cinqesmilimes')
os.mkdir(has_tumor_tissue)


# create new folders inside val_dir
no_tumor_tissue = os.path.join(val_dir, 'dinar')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'deuxdinars')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'cinqdinars')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'dixmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'vingtmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'cinqentesmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'centmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'deuxcentsmilimes')
os.mkdir(has_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'cinqentesmilimes')
os.mkdir(has_tumor_tissue)


# In[ ]:


os.listdir('base_dir/train_dir')


# In[ ]:


def Image_read(image):
    img = cv2.imread(image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return x
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob,cv2
imgList = glob.glob("../input/trainimages/train/train/*")
img = Image_read("../input/trainimages/train/train/559210.jpg")
print(img)
plt.figure()
plt.imshow(img) 
plt.show() 
minx=0
maxx=224
miny=0
maxy=224
test=True
testy=True
for i in range(224):
    for j in range(224):
        if img[i][j]<100:
            if(test==True):
                minx=j
                test=False
            if(j<maxx):
                maxx=j
            if(testy==True):
                miny=i
                testy=False
            if(i<maxy):
                maxy=i
print(minx,maxx,miny,maxy)
retval, thresh_gray = cv2.threshold(img, thresh=150, maxval=255,type=cv2.THRESH_BINARY_INV)
plt.figure()
plt.imshow(thresh_gray) 
plt.show() 
contours,h = cv2.findContours(thresh_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE   )
for cnt in contours:
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    print(center,radius)

cv2.circle(img,(164,112),125,(0,255,0),2)
plt.figure()
plt.imshow(img) 
plt.show()             
            


# In[ ]:


import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  os # data 

# Image processing
from PIL import Image, ImageFile
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
X_data = []
Y_data = []
imgList = glob.glob("../input/trainimages/train/*.png")
data=pd.read_csv("../input/pices-de-monnaie/train.csv")


# In[ ]:


print(list(data[data.img==725416].label)[0])
print(data.shape[0])


# In[ ]:


folder = "../input/trainimages/train/train/"
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
imgList = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} images".format(len(imgList)))
print("Image examples: ")
for i in range(40, 42):
    print(imgList[i])
    display(_Imgdis(filename=folder + "/" + imgList[i], width=224, height=224))


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm


# In[ ]:


train_image = []
folder = "../input/trainimages/train/train/"
i=0
while(i < data.shape[0]-1):
    if(data['img'][i]!=997180):
        img = image.load_img(folder+data['img'][i].astype('str')+'.jpg', target_size=(28,28,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y=data['label'][i]
        y = to_categorical(y)
    i+=1
X = np.array(train_image)


# In[ ]:


for i in tqdm(range(data.shape[0])):
    if(data['img'][i]!=997180):
        y=data['label'].values
        y = to_categorical(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[ ]:


# Original Dimensions
image_width = 224
image_height = 224
ratio = 2

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1
X_data=[]
Y_data=[]
X_data = np.ndarray(shape=(len(imgList), image_height, image_width, channels),
                     dtype=np.float32)

i = 0
k=imgList[:100]
for im in k:
    img = load_img(folder + "/" + im)  # this is a PIL image
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((112, 112,3))
    # Normalize
    x = (x - 128.0) / 128.0
    if list(data[data.img==int(im[:-4])].label):
        X_data[i] = x
        Y_data.append (list(data[data.img==int(im[:-4])].label)[0])
        i+=1


# In[ ]:


from skimage import io, transform
def import_data():
    img_data = []
    label_data = []
    
    for img in imgList:
        img_read = io.imread(folder + "/" +img)
        img_read = transform.resize(img_read, (128,128), mode = 'constant')
        img_data.append(img_read)
        label_data.append(list(data[data.img==int(im[:-4])].label)[0])
        
    return train_test_split(np.array(img_data)[:100], np.array(label_data)[:100], test_size=0.2, random_state=33)


# In[ ]:


X_train, X_test, y_train, y_test= import_data()


# In[ ]:


from sklearn.model_selection import train_test_split

#Splitting 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=33)
print("Train set size: {0}, Val set size: {1}".format(len(X_train),len(X_test)))


# In[ ]:


no_of_classes = len(np.unique(y_train))
print(no_of_classes)
print(X_data[0])
print(X_data[0][0].size)
for i in range(100):
    print(Y_data[i])


# In[ ]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,no_of_classes,dtype='float32')
y_test = np_utils.to_categorical(y_test,no_of_classes,dtype='float32')


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(100,100,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(9,activation = 'softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print('Compiled!')


# In[ ]:


batch_size = 32

checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)

history = model.fit(X_train,y_train,
        batch_size = 32,
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)


# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import PIL
from keras import layers
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
import keras.backend as K
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


os.listdir("../input/")


# In[ ]:


data_df = pd.read_csv("../input/pices-de-monnaie/train.csv")
data_df['label'].value_counts()


# In[ ]:


y=data_df[data_df['img'] != 997180]['label']
data_df=data_df[data_df['img'] != 997180]
print(data_df.shape)


# In[ ]:


def prepareImages(data, m):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
        
    for fig in data['img']:
        try:
            #load images into images of size 100x100x3
            img = image.load_img("../input/trainimages/train/train/"+str(fig)+".jpg", target_size=(100, 100, 3))
            x = image.img_to_array(img)
            x = preprocess_input(x)

            X_train[count] = x
            if (count%500 == 0):
                print("Processing image: ", count+1, ", ", fig)
            count += 1
        except:
            print(fig,'.jpg is not found')
            
    return X_train


# In[ ]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder


# In[ ]:


X = prepareImages(data_df, data_df.shape[0])
X /= 255


# In[ ]:


y, label_encoder = prepare_labels(y)
y.shape


# In[ ]:


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# In[ ]:


model = vgg19.VGG19(input_shape=(32, 32, 3), weights=None, classes=9)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())


# In[ ]:


X_train=[]
y_train=[]
X_test=[]
y_test=[]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[ ]:


batch_size = 100

checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)

history = model.fit(X_train,y_train,
        batch_size = 100,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)


# In[ ]:


history = model.fit(X_train, y_train, epochs=24, batch_size=24, verbose=1)

