#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > > **Importing the suitable libraries : **

# In[ ]:


from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import warnings
from sklearn.externals import joblib
warnings.filterwarnings("ignore")
#warnings.FutureWarning("ignore")
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[ ]:


os.listdir(os.getcwd())


# In[ ]:


os.getcwd()


# In[ ]:


data = []
labels= []
data_1=os.listdir("../input/cell_images/cell_images/Parasitized/")


# In[ ]:


for i in data_1:
    try:
        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)
        image_from_array= Image.fromarray(image , "RGB")
        size_image =image_from_array.resize((50,50))
        #resize45=size_image.rotate(15)
        #resize75 = size_image.rotate(25)
        #blur =cv2.blur(np.array(size_image),(10,10))
        data.append(np.array(size_image))
        labels.append(0)
        #labels.append(0)
        #labels.append(0)
        #labels.append(0)
        
    except AttributeError:
        print("")
Uninfected = os.listdir("../input/cell_images/cell_images/Uninfected/")
for b in Uninfected:
    try :
        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)
        array_image=Image.fromarray(image,"RGB")
        size_image=array_image.resize((50,50))
        resize45= size_image.rotate(15)
        resize75 = size_image.rotate(25)
        #blur =cv2.blur(np.array(size_image),(10,10))
        data.append(np.array(size_image))
        #data.append(np.array(resize45))
        #data.append(np.array(resize75))
        #data.append(np.array(blur))
        #labels.append(1)
        #labels.append(1)
        #labels.append(1)
        labels.append(1)
    except AttributeError:
        print("")


# In[ ]:


Cells =np.array(data)
labels =np.array(labels)


# In[ ]:


print(labels.shape)
print(Cells.shape)


# In[ ]:


#np.save("Cells_data",Cells)
#np.save("labels_data",Cells)


# In[ ]:


#Cells =np.load("Cells_data.npy")
#labela =np.load("labels_data.npy")


# In[ ]:


s=np.arange(Cells.shape[0])


# In[ ]:


np.random.shuffle(s)


# In[ ]:


len_data = len(Cells)


# In[ ]:


Cells=Cells[s]
labels =labels[s]


# In[ ]:


labels =keras.utils.to_categorical(labels)


# In[ ]:


model =Sequential()


# In[ ]:


model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()


# In[ ]:


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[ ]:


Cells=Cells/255


# In[ ]:


model.fit(Cells,labels,batch_size=50,epochs=10,verbose=1)


# In[ ]:


model.save("my_model.h5")


# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0')

import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:


tf.lite.TFLiteConverter.


# In[ ]:


import tensorflow as tf


# In[ ]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# In[ ]:


converter = tf.lite.TFLiteConverter.from_saved_model("../input/my_model.h5")
tflite_model = converter.convert()


# In[ ]:


tf.__version__


# In[ ]:





# In[ ]:





# In[ ]:


joblib.dump(model,"model")


# In[ ]:


joblib.load("model")
joblib.dump(model,"model")


# In[ ]:


model.save("model111.h5")


# In[ ]:


from keras.models import load_model
model11=load_model("model111.h5")


# In[ ]:


model11.predict(Cells[73].reshape(1,50,50,3))


# In[ ]:


blur=cv2.blur(Cells[1000].rotate(45),(5,5))


# In[ ]:


plt.imshow(blur)


# In[ ]:


plt.plot(histroy.history["loss"])#.keys()


# In[ ]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(model,"Malaria Cell model")


# In[ ]:


from keras.applications.xception import Xception


# In[ ]:


model1=Xception()


# In[ ]:


modl= keras.applications.vgg16.VGG16()


# In[ ]:


modl.summary()


# In[ ]:


from keras.applications import VGG16 
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_img=ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.2,horizontal_flip=True)


# In[ ]:


train_images=train_img.flow_from_directory("../input/cell_images/cell_images/Parasitized/",target_size=(64,64,3),batch_size=32)

