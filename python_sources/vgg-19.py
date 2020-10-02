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


# In[ ]:


import pandas as pd 
import cv2     
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage 
from skimage import filters
from skimage.transform import resize
from matplotlib import pyplot as plt
print(os.listdir("../input/chest_xray"))


# In[ ]:


Train="../input/chest_xray/chest_xray/train/"
Test="../input/chest_xray/chest_xray/test/"


# In[ ]:


img = cv2.imread("../input/chest_xray/chest_xray/train/PNEUMONIA/person755_bacteria_2659.jpeg")

plt.imshow(img)
c = img[:,:,0] 
d = img[:,:,1]
e = img[:,:,2]
x=0
y=0
j1 =int(c.mean())
j2 =int(d.mean())
j3 =int(e.mean())
               
print(j1)
print(j2)
print(j3)

rgb = cv2.inRange(img, (j1,j2,j3), (255, 255, 255))
rgb1= filters.sobel(rgb)
img = skimage.transform.resize(img, (150, 150, 3))

plt.imshow(rgb)
plt.imshow(rgb1)


# In[ ]:


plt.imshow(rgb)


# In[ ]:


img = cv2.imread("../input/chest_xray/chest_xray/train/PNEUMONIA/person755_bacteria_2659.jpeg")
plt.imshow(img)


# In[ ]:


dst = cv2.addWeighted(rgb,0.5,rgb,0.5,0)
plt.imshow(dst)


# In[ ]:



                img = cv2.imread("../input/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0927-0001.jpeg")
                
                plt.imshow(img)
                c = img[:,:,0] 
                d = img[:,:,1]
                e = img[:,:,2]
                x=0
                y=0
                j1 =int(c.mean())
                j2 =int(d.mean())
                j3 =int(e.mean())
               
                print(j1)
                print(j2)
                print(j3)
                
                rgb = cv2.inRange(img, (j1,j2,j3), (255, 255, 255))
                
                img = skimage.transform.resize(img, (150, 150, 3))
                plt.imshow(rgb)
                


# In[ ]:


from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt


# In[ ]:


IMAGE_SIZE =[224,224]
Train ="../input/chest_xray/train"
Test ="../input/chest_xray/test"


# In[ ]:


vgg = VGG19(input_shape =IMAGE_SIZE+[3],weights='imagenet',include_top = False )


# In[ ]:


for layer in vgg.layers:
    layer.trainable = False
    


# In[ ]:


folder = glob("../input/chest_xray/chest_xray/train")
print(folder)


# In[ ]:


import os
print(os.listdir("../input/chest_xray/chest_xray/train"))


# In[ ]:


x = Flatten()(vgg.output)


# In[ ]:


predition = Dense(2,activation = 'softmax')(x)
model =Model(inputs =vgg.input,outputs =predition)


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png',show_layer_names =True,show_shapes = True)


# In[ ]:


import numpy as np
img1 = "../input/chest_xray/chest_xray/test/NORMAL/IM-0011-0001-0001.jpeg"
img1 = image.load_img(img1, target_size=(224, 224))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)

features1 = model.predict(x1)


# In[ ]:


print(features1)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255) 


# In[ ]:


import os
print(os.listdir("../input/chest_xray/train"))


# In[ ]:


train_generator = train_datagen.flow_from_directory( 
    Train, 
    target_size=(224, 224), 
    batch_size=500, 
    class_mode='categorical') 
  
validation_generator = test_datagen.flow_from_directory( 
    Test, 
    target_size=(224, 224), 
    batch_size=500, 
    class_mode='categorical') 


# In[ ]:


history = model.fit_generator(train_generator,
                       validation_data = validation_generator,
                       epochs =6,
                       steps_per_epoch = 40,
                       validation_steps = 10)


# In[ ]:


import matplotlib.pyplot as plt
img1 = "../input/chest_xray/chest_xray/test/NORMAL/NORMAL2-IM-0381-0001.jpeg"
img1 = image.load_img(img1, target_size=(224, 224))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)

features1 = model.predict(x1)


# In[ ]:


print(features1)
print(np.round(features1))


# In[ ]:


img = "../input/chest_xray/chest_xray/test/PNEUMONIA/person1685_virus_2903.jpeg"
img = image.load_img(img, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)


# In[ ]:


print(features)
print(np.round(features))


# In[ ]:


model.save_weights('vgg19.h5')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix


# In[ ]:


text_img,text_label = next(validation_generator)
text_label = text_label[:,0]
text_label


# In[ ]:


prediction = model.predict_generator(validation_generator, steps=1, verbose=0)
cm = confusion_matrix(text_label,np.round(prediction[:,0]))
cm_plot_label = ['Normal','Pneumonia']
plot_confusion_matrix(cm,cm_plot_label)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


score  = model.evaluate(validation_generator)
print(score[1])


# In[ ]:


from sklearn.metrics import accuracy_score
acc =accuracy_score(text_label,np.round(prediction[:,0]))
print(acc)

