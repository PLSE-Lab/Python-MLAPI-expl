#!/usr/bin/env python
# coding: utf-8

# This kernel is my first try at making a NN using keras to apply it to the cancer cell competion. Any comments are more than welcome on any topic, as I am a very early beginner in data science :-)

# **Things that I learned**
# This is my first deep learning code, so obviously, it can only be a learning experience. Since I've read a few Kernels already, I figured if any other beginner like me stumbles upon my Kernel, maybe this might be helpful. If not, well at least I get to write feedback for myself ;-)
# 
# - I started by trying to make a [57k, 96, 96, 3] np.ndarray containing all the arrays of all the images we need to classify. While that did seem to work with smaller set (I tried with 25k, and it worked), at 57k, the Kernel just crashes. After some investigation (*puts Sherlock's hat on*) the issue seems to be memory overload. I mean I'm just turning 57,000+ images into 96x96x3 arrays, what could go wrong? Next step is to try inserting the prediction inside the for loop. Here's the idea: I'm still training my model (Well, not really mine, rather the one used in the week 2 of Course 4 of Andrew Ng's Coursera Deep Learning course) with a small amount of data, just to see if it's working. I'm taking baby steps, I'll gradually add more data as things work (eventually, I hope). I saw a tweet from Andrej Karapathy a fw weeks ago saying that you should try making a small model, with little data, until it overfits, and then move on. This allows you to check that the model is working, and helps find potential (as a beginner, I'd change the word 'potential' by 'numerous', but maybe it's just me) sources of error. Once I have my first model, then the prediciton is done image by image: I take an image, convert it to an array, and then predict it. Repeat ~57,000 times. This means no storing of the arrays, and (hopefully) no memory crash.
# 
# - I did the above, successfully. Meaning that I can get an output and send it for classification (I got 0.6520 with about 2,000 training examples, 4 epochs; and this grew to 0.6793 with 25,000 training examples and 20 epochs). So next step, is to change the model, or neural network. It seems like I am getting only slight improvements with adding more training examples, which is good, but I am sure I can make bigger steps, without using more than a few thousand training examples. Then, once I found a much better model, I will add more examples, and training epochs. Next step: trying the Keras build-in ResNet50.

# **TO DO**
# Problem at the moment: memory overload.
# - Train model with about 2000 train ex, 500 test ex, and then do the prediction inside the loop, so no array storing of the image on submission set. In submission loop: img_to_array, then predict, then put prediction into sample_submission['label']. I think this is the file that has to be submitted again, have to check that.
# 
# - I have taken the predicitons and made that any prediction >= 0.5 should be considered as 1, and any prediciton < 0.5 should be labelled as 0. I don't know if that 0.5 limit is good, or if it should be higher/lower?
# 
# - The competition says that the goal of the NN is to determine if there are cancer pixels in the 32x32 main area of the image. One idea would be to crop the images down to the 32x32 main part, and run the model & estimate, to see if it runs better like that.
# 
# - I am using an Adam optimizer, with binary crossentropy loss. This might be interesting to look into aswell

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import time

import os
print(os.listdir("../input"))


# In[ ]:


train_labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
test_labels = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

#print('train : ','\n', train_sample.head(5))
#print('test : ','\n', test_labels.head(5))


# In[ ]:


test_labels.shape


# Very early remarks:
# 
# 1. We have 220,025 images in train_labels, of which 89,117 are labelled as having a cancer pixel in the 32x32 center zone.
# 2. There are 57,5k images in sample_sumbissions
# 3. Images are of size 96x96x3 (Meaning RBG, and of total size 27,648)
# 4. The only information we have is, if there is a cancer cell of not (in the 32x32 center zone, according to the data description). We do not know what it looks like, which pixel is identified as the one being the cancer, and what makes or doesn't make a cancer cell. This will make EDA relatively fast in my opinion, because there isn't much information we are going to be abel to look at; apart from looking at 1 labelled pictures, and visually trying to find what looks like the patterns compared to 0 labelled data.
# 

# In[ ]:


#This image is labelled as having a cancer cell.
image = plt.imread('../input/histopathologic-cancer-detection/train/c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')
plt.imshow(image)
plt.show()


# In[ ]:


image.shape


# Now we are going to try (emphasis on the word 'try') to make a model based on the exciting keras models

# In[ ]:


#let's start with a small sample first:

#size train sample:
x = 30000

#size of val sample:
l = 5000

train_sample = train_labels[:x]
val_sample = train_labels[x:x+l]
test_sample = test_labels


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.utils import layer_utils, to_categorical
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


img_heigth, img_width = 96, 96


# In[ ]:


img=load_img('../input/histopathologic-cancer-detection/train/c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')


# In[ ]:


train_sample.iloc[0][1]


# In[ ]:


nb_train_examples=train_sample.shape[0]
nb_val_examples=val_sample.shape[0]

train_img_array = np.ndarray(shape=[nb_train_examples, 96, 96, 3])
train_img_label = np.ndarray(shape=[nb_train_examples, 1])

val_img_array = np.ndarray(shape=[nb_val_examples, 96, 96, 3])
val_img_label = np.ndarray(shape=[nb_val_examples, 1])

test_img_array = np.ndarray(shape=[test_sample.shape[0], 96, 96, 3])
test_img_label = np.ndarray(shape=[test_sample.shape[0], 1])


# In[ ]:


t1=time.time()
for p in range(nb_train_examples):
    #We turn the .tif into an array
    img_name=train_sample.iloc[p][0]
    img=load_img('../input/histopathologic-cancer-detection/train/'+img_name+'.tif')
    img=img_to_array(img)
    img=img/255
    #print(img_name)
    #print(img.shape)
    train_img_array[p]=img #putting the image inside the 4 dim array
    
    #We put the label into a new ndarray:
    train_img_label[p]=train_sample.iloc[p][1]
t2=time.time()
print('time to turn .tif into array for train_set : ',t2-t1)
print('train_img_array shape is : ', train_img_array.shape)
print('train_img_label shape is : ', train_img_label.shape)


# In[ ]:


t1=time.time()
for p in range(nb_val_examples):
    #We turn the .tif into an array
    img_name=val_sample.iloc[p][0]
    img=load_img('../input/histopathologic-cancer-detection/train/'+img_name+'.tif')
    img=img_to_array(img)
    img=img/255
    #print(img_name)
    #print(img.shape)
    val_img_array[p]=img #putting the image inside the 4 dim array
    
    #We put the label into a new ndarray:
    val_img_label[p]=val_sample.iloc[p][1]
t2=time.time()
print('time to turn .tif into array for val_sample : ',t2-t1)
print('val_img_array shape is : ', val_img_array.shape)
print('val_img_label shape is : ', val_img_label.shape)


# It takes about 2.42 seconds to do img_to_array on 1,000 examples. This means it's going to take about 8.4 min for all 220k training examples, + 2,3 min for the test set. That's about 11 min in total just to convert the images to numpy using this method. (goes much much faster whe using GPU)
# 
# It's long, but at least it works, so for now, that's good enough.

# In[ ]:


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def FirstModel(num_classes):
    
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling='avg'))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    
    model.layers[0].trainable = False
    
    return model


# In[ ]:


#my_model=FirstModel(train_img_array[0].shape)
my_model = FirstModel(num_classes = 2)


# In[ ]:


my_model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy', auroc])


# In[ ]:


train_img_label = to_categorical(train_img_label, num_classes=2)
val_img_label = to_categorical(val_img_label, num_classes=2)


# In[ ]:


stats = my_model.fit(x = train_img_array, y = train_img_label, epochs = 5)


# In[ ]:


evaluation = my_model.evaluate(x= val_img_array, y=val_img_label)
print()
print ("Loss = " + str(evaluation[0]))
print ("Test Accuracy = " + str(evaluation[1]))


# In[ ]:


#This turned out to be a bad idea. But I'm keeping it, never know when I might need it

dummy_img = np.ndarray(shape=(1, 96, 96, 3))

t1=time.time()
for p in range(test_sample.shape[0]):
    #We turn the .tif into an array
    img_name=test_sample.iloc[p][0]
    img=load_img('../input/histopathologic-cancer-detection/test/'+img_name+'.tif')
    img=img_to_array(img)
    img=img/255
    
    #print(img_name)
    #print(img.shape)
    
    pred = my_model.predict(img.reshape(1,96,96,3))
    #We put the label into a new ndarray:
    test_sample.at[p ,'label'] = pred.argmax()
    
    
t2=time.time()
print('time to turn .tif into array for test_sample : ',t2-t1)


# In[ ]:


print('number of images labelled with cancer : ',test_sample[test_sample['label']==1].shape[0],
      ' out of ', test_sample.shape[0], ' examples')


# In[ ]:


test_sample.to_csv('test_predictions.csv', index=False)

