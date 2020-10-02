#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# using new cropped oct because it was not cropped to 60x300 and has original dimensions
#batch norm not being trained

import os
import keras
from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.utils import Sequence
from cv2 import * #Import functions from OpenCV
import cv2
import glob
from skimage.transform import resize
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.models import model_from_json
import json
from statistics import mean


# In[ ]:


img = glob.glob("../input/no-med-filt-224/no_median_filter_224/no_median_filter_224/train/CNV/*.jpeg");
img = img + glob.glob("../input/no-med-filt-224/no_median_filter_224/no_median_filter_224/train/DME/*.jpeg");
l = len(img);
y = np.zeros((l,3))
y[:,2] =1
img = img + glob.glob("../input/no-med-filt-224/no_median_filter_224/no_median_filter_224/train/DRUSEN/*.jpeg");
m = len(img);
k = np.zeros((m-l,3));
k[:,1] = 1;
y = np.append(y,k, axis =0);
img = img + glob.glob("../input/no-med-filt-224/no_median_filter_224/no_median_filter_224/train/NORMAL/*.jpeg");
k = np.zeros((len(img)-m,3));
k[:,0] = 1;
y = np.append(y,k, axis =0);
from sklearn.utils import shuffle
img, y = shuffle(img,y)
img, y = shuffle(img,y)
img, y = shuffle(img,y)
img, y = shuffle(img,y)
img, y = shuffle(img,y)

from sklearn.model_selection import KFold
## Training with K-fold cross validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf.get_n_splits(img)

X = np.array(img)
y = np.array(y)
np.savetxt('X.out',X,fmt='%s')
np.savetxt('y.out',y)
i=1
for train_index, test_index in kf.split(img):
    trainData = X[train_index]
    testData = X[test_index]
    trainLabels = y[train_index]
    testLabels = y[test_index]
    np.savetxt('train_index-'+str(i)+'.out',train_index)
    np.savetxt('test_index-'+str(i)+'.out',test_index)
    i+=1

