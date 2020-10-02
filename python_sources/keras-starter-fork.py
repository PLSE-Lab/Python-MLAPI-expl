#!/usr/bin/env python
# coding: utf-8

# ### This is a basic Keras model. 

# In[ ]:


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

trainpath = '../input/train/'
testpath = '../input/test/'

print('# of training files: ' + str(len(os.listdir(trainpath))))
print('# of testing files: ' + str(len(os.listdir(testpath))))


# In[ ]:


# Preview labels
train_labels = pd.read_csv('../input/train_labels.csv')
print(train_labels.head())


# In[ ]:


# Preview a few noninvasive plant images
from skimage import io, transform
import matplotlib.pyplot as plt

for i in range(12):
#    print(train_labels.name[i], train_labels.invasive[i])
    sample_image = io.imread(trainpath + str(train_labels.name[i]) + '.jpg')
    if train_labels.invasive[i] < 1:
        print('Non-Invasive image example' + trainpath + str(train_labels.name[i]) + '.jpg (Height:{0} Width:{1})'.format(sample_image.shape[0], sample_image.shape[1]))
        plt.figure()
        plt.imshow(sample_image)


# In[ ]:


# Preview a few invasive plant images

for i in range(12):
#    print(train_labels.name[i], train_labels.invasive[i])
    sample_image = io.imread(trainpath + str(train_labels.name[i]) + '.jpg')
    if train_labels.invasive[i] > 0:
        print('Invasive image example' + trainpath + str(train_labels.name[i]) + '.jpg (Height:{0} Width:{1})'.format(sample_image.shape[0], sample_image.shape[1]))
        plt.figure()
        plt.imshow(sample_image)


# In[ ]:


import cv2
# There is one image in the test set that has different dimensions.
# It may just need a rotation, but I'm going to ignore it for now.
imhei = 866
imwid = 1154

for i in range(1060,1070):
    print(train_labels.name[i], train_labels.invasive[i], sample_image.shape)
    sample_image = io.imread(testpath + str(train_labels.name[i]) + '.jpg')
    if sample_image.shape[0] == imhei and sample_image.shape[1] == imwid:
        continue
    rrows,ccols = sample_image.shape[0:2]
    M = cv2.getRotationMatrix2D((ccols/2,rrows/2),90,1)
    sample_image = cv2.warpAffine(sample_image,M,(ccols,rrows))
    sample_image.reshape(ccols,rrows,3)
    if train_labels.invasive[i] > 0:
        print('Invasive image example' + testpath + str(train_labels.name[i]) + '.jpg (Height:{0} Width:{1})'.format(sample_image.shape[0], sample_image.shape[1]))
        plt.figure()
        plt.imshow(sample_image)
    if train_labels.invasive[i] < 1:
        print('Non-Invasive image example' + testpath + str(train_labels.name[i]) + '.jpg (Height:{0} Width:{1})'.format(sample_image.shape[0], sample_image.shape[1]))
        plt.figure()
        plt.imshow(sample_image)


# In[ ]:


# Check that input_shape = (batch_size, rows, columns, channels)
from keras.backend import image_data_format
print(image_data_format())


# In[ ]:


# Kernel memory is limited so I'm using 100 images each for training and validation 
# and scaling them down to 150x200 pixels to keep things simple.
xpix = 150
ypix = 200
ncol = 3
scaled = (xpix, ypix, ncol)
n_train = 100
x_train = np.empty(shape=(n_train, xpix, ypix, ncol))
y_train = np.array(train_labels.invasive.values[0:n_train])
n_eval = 50
x_val = np.empty(shape=(n_eval, xpix, ypix, ncol))
y_val = np.array(train_labels.invasive.values[n_train:n_train+n_eval])

for i in range(n_train):
    tr_im = io.imread(trainpath + str(i+1) + '.jpg')
    x_train[i] = transform.resize(tr_im, output_shape=scaled)

for i in range(n_eval):
    val_im = io.imread(trainpath + str(i+n_train+1) + '.jpg')
    x_val[i] = transform.resize(val_im, output_shape=scaled)


# In[ ]:


# Starting architecture
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(150, 200, 3)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(256, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=SGD(lr=1e-5, momentum=0.75, nesterov=False), 
              loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Look at how tensors affect output shape
print(model.summary())


# In[ ]:


# One epoch for demonstration purposes
model.fit(x_train, y_train, epochs=1, batch_size=50)


# In[ ]:


acc = model.evaluate(x_val, y_val)[1]
print('Evaluation accuracy:{0}'.format(round(acc, 4)))


# In[ ]:


# reading test sample
sample_submission = pd.read_csv("../input/sample_submission.csv")
img_path = "../input/test/"

test_names = []
file_paths = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.ix[i][0])
    file_paths.append( img_path + str(int(sample_submission.ix[i][0])) +'.jpg' )
    
test_names = np.array(test_names)


# In[ ]:


test_images = []
for file_path in file_paths:
    #read image
    img = io.imread(file_path)
    img = transform.resize(img, output_shape=scaled)
    test_images.append(img)
    
    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)


# In[ ]:


predictions = model.predict(test_images)

for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("submit.csv", index=False)

