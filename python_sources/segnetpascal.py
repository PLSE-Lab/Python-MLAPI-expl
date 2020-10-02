#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://github.com/0bserver07/Keras-SegNet-Basic/blob/master/Segnet-Evaluation-Visualization.ipynb
#https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
#https://github.com/divamgupta/image-segmentation-keras/blob/master/visualizeDataset.py
#https://www.kaggle.com/llawliet426/segnetpascal/edit


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
#print(os.listdir("../input/voc2012/VOC2012/SegmentationClass"))
#SC= os.listdir("../input/voc2012/VOC2012/SegmentationClass")


#DATASET=os.listdir("../input/voc2012/VOC2012")
#print(os.listdir("../input/voc2012/VOC2012/JPEGImages"))
#DATASET=os.listdir("../input/voc2012/VOC2012/ImageSets/Main")
#print(DATASET)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import img_to_array
#image = mpimg.imread("../input/voc2012/VOC2012/JPEGImages/2010_007828.jpg")
#plt.imshow(image)
#plt.show()


#file1 = open("../input/voc2012/VOC2012/ImageSets/Main/aeroplane_train.txt","r") 
#print"Output of Readlines after appending"
#print(file1.readlines()) 
#file1.close() 


#os.path.join(DATASET, '/train.txt')
# Any results you write to the current directory are saved as output.


# In[ ]:


image = mpimg.imread("../input/voc2012/VOC2012/SegmentationClass/2009_000420.png")
plt.imshow(image)
plt.show()
image = mpimg.imread("../input/voc2012/VOC2012/SegmentationObject/2009_000420.png")
plt.imshow(image)
plt.show()
print(image.shape)


# In[ ]:


from scipy import misc


# In[ ]:


def load_data(original_img_path,original_mask_path):
    original_img = cv2.imread(original_img_path)[:, :, ::-1]
    
    array_img = img_to_array(original_img)/255
    #image=array_img.reshape(330,500,3)
    original_img=misc.imresize(original_img, (330,500,3))
    original_mask = cv2.imread(original_mask_path)
    print(original_img.shape)
    


# In[ ]:


Dataset=os.listdir("../input/voc2012/VOC2012/SegmentationClass")
#os.listdir("../input/voc2012/VOC2012/JPEGImages")
count=0
datafile=[]
for i in  Dataset:
    #print(i)
    #print(i.split('.',1)[0])
    if count>10:
        break;
    nstr="../input/voc2012/VOC2012/JPEGImages/"+i.split('.',1)[0]+".jpg"
    #print(nstr)
    exists = os.path.isfile(nstr)
    if exists:
        #print(nstr)
        pstr="../input/voc2012/VOC2012/SegmentationClass/"+i
        
        image = mpimg.imread(pstr)
        
        plt.imshow(image)
        plt.show()
        load_data(nstr,pstr)
        image = mpimg.imread(nstr)
        plt.imshow(image)
        plt.show()
        datafile.append(i)
        count=count+1
    else:
        print("not found {}".format(nstr))
#print(datafile)


# In[ ]:



    def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]




segnet_basic = models.Sequential()
#330, 500, 3
segnet_basic.add(Layer(input_shape=(3,330, 500)))

segnet_basic.encoding_layers = create_encoding_layers()
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)

# Note: it this looks weird, that is because of adding Each Layer using that for loop
# instead of re-writting mode.add(somelayer+params) everytime.

segnet_basic.decoding_layers = create_decoding_layers()
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)

segnet_basic.add(Convolution2D(12, 1, 1, border_mode='valid',))

segnet_basic.add(Reshape((12,data_shape), input_shape=(12,330, 500)))
segnet_basic.add(Permute((2, 1)))
segnet_basic.add(Activation('softmax'))
    
    


# In[ ]:


for i in DATASET:
    load_data(i);


# In[ ]:




