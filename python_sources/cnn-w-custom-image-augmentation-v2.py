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


# ## References:
# https://www.kaggle.com/myltykritik/simple-lgbm-image-features
# 
# https://stackoverflow.com/questions/47200146/keras-load-images-batch-wise-for-large-dataset
# 
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# 
# https://keras.io/models/model/
# 
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# 
# https://keras.io/layers/pooling/
# 
# https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649
# 

# In[ ]:


# !ls ../input/train_images


# ## Imports

# In[ ]:


# time
import time
# import OpenCV
import cv2
# random
import random
random.seed(1331)
# viz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2
conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


conv_base.summary()


# ## Get as sense of the images shapes

# In[ ]:


print(cv2.imread('../input/train_images/8132fabfd-10.jpg').shape)
print(cv2.imread('../input/train_images/beb0f98f7-1.jpg').shape)
print(cv2.imread('../input/train_images/d168549f2-4.jpg').shape)
print(cv2.imread('../input/train_images/9d6c6055b-2.jpg').shape)
print(cv2.imread('../input/train_images/a6593fb48-6.jpg').shape)
print(cv2.imread('../input/train_images/ea1736eec-4.jpg').shape)
print(cv2.imread('../input/train_images/f205b0649-6.jpg').shape)
print(cv2.imread('../input/train_images/f888d5f54-3.jpg').shape)
print(cv2.imread('../input/train_images/bcf546cb8-3.jpg').shape)
print(cv2.imread('../input/train_images/3fd545213-6.jpg').shape)


# ## View an Image

# In[ ]:


# read in image
img = cv2.imread('../input/train_images/0008c5398-1.jpg')
img = cv2.resize(img, (224,224))
# show image
plt.imshow(img)


# In[ ]:


np.max(img)


# # Setting up PetID and AdoptionSpeed Lookup Table
# 
# - So we want to create a table where we can call the ID# and get the target values
# - We'll also test that we can grab the target value via the ID#

# In[ ]:


# read in training dataset
train_df = pd.read_csv('../input/train/train.csv')
# only grab the ID and target
pet_ids_and_target = train_df[['PetID','AdoptionSpeed']]
# Get one-hot representation using get_dummies
onehottarget = pd.get_dummies(pet_ids_and_target['AdoptionSpeed'])
# merge one-hot encodings back to our table
pet_ids_and_target = pet_ids_and_target.join(onehottarget)
# show our final table
print(pet_ids_and_target.head())


# In[ ]:


# test out AdoptionSpeed extraction code
getrow = pet_ids_and_target.loc[pet_ids_and_target['PetID'] == 'bec2fe7ad']
print(getrow)
print(np.asarray(getrow[[0,1,2,3,4]].values[0], dtype=np.float32))


# ## My Data Augmentation Function
# 
# - Here we're creating our own custom image augmentation code using OpenCV
# - We'll use this function to augment our images during training
# - With each batch we'll generate a random number which will indicate which augmentation we do.  We pick random augmentations with each batch mainly due to memory constraints.
# - You can customize this to include any augmentations that you want!  Change the random # range, include a new elif or alter the current elif's as you want!
# - You might want to do this if you have some contexual knowledge about the images you'd like to leverage but need more control over the augmentations.
# - Here we're mainly doing specific cropping, rotating, and blurring techniques based on knowledge of the images.

# In[ ]:


def MyAug(fiximg, shape):
    auglist = [fiximg]
    
    # we can only fit so much into memory
    # so every round we will randomly select an augmentation to apply for training
    augshuf = random.randint(1,101)
    
    if augshuf in range(1,11):
    
        # Augmentation 1

        # blur image using 25x25 kernel
        blurred = cv2.blur(fiximg, ksize=(25,25))
        # write out image
        auglist.append(blurred)
    
    elif augshuf in range(11,21):
    
        # Augmentation 2

        # covert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # We're going to equalize the value channel
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        # convert back to RGB
        eq_color_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # write out image
        auglist.append(eq_color_img)
    
    elif augshuf in range(21,31):
    
        # Augmentation 3

        y_offsetp1 = int(fiximg.shape[0]*0.25)
        y_offsetp2 = int(fiximg.shape[0]*0.75)
        x_offsetp1 = int(fiximg.shape[1]*0.25)
        x_offsetp2 = int(fiximg.shape[1]*0.75)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)
    
    elif augshuf in range(31,41):
    
        # Augmentation 4

        y_offsetp1 = int(fiximg.shape[0]*0)
        y_offsetp2 = int(fiximg.shape[0]*0.6)
        x_offsetp1 = int(fiximg.shape[1]*0)
        x_offsetp2 = int(fiximg.shape[1]*0.6)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)

    elif augshuf in range(41,51):
        
        # Augmentation 5

        y_offsetp1 = int(fiximg.shape[0]*0)
        y_offsetp2 = int(fiximg.shape[0]*0.6)
        x_offsetp1 = int(fiximg.shape[1]*0.4)
        x_offsetp2 = int(fiximg.shape[1]*1)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)
    
    elif augshuf in range(51,61):
    
        # Augmentation 6

        y_offsetp1 = int(fiximg.shape[0]*0)
        y_offsetp2 = int(fiximg.shape[0]*0.6)
        x_offsetp1 = int(fiximg.shape[1]*0.4)
        x_offsetp2 = int(fiximg.shape[1]*1)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)
    
    elif augshuf in range(61,71):
    
        # Augmentation 7
    
        y_offsetp1 = int(fiximg.shape[0]*0.4)
        y_offsetp2 = int(fiximg.shape[0]*1)
        x_offsetp1 = int(fiximg.shape[1]*0.4)
        x_offsetp2 = int(fiximg.shape[1]*1)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)
    
    elif augshuf in range(71,81):
    
        # Augmentation 8

        xp1 = min(.15,random.randint(1,100)/100)
        xp2 = max(.85,random.randint(1,100)/100)
        yp1 = min(.15,random.randint(1,100)/100)
        yp2 = max(.85,random.randint(1,100)/100)
        # create points
        y_offsetp1 = int(fiximg.shape[0]*yp1)
        y_offsetp2 = int(fiximg.shape[0]*yp2)
        x_offsetp1 = int(fiximg.shape[1]*xp1)
        x_offsetp2 = int(fiximg.shape[1]*xp2)
        # crop
        cropped_img = fiximg[y_offsetp1:y_offsetp2,x_offsetp1:x_offsetp2]
        cropped_img = cv2.resize(cropped_img, shape)
        # write out image
        auglist.append(cropped_img)
        
    elif augshuf in range(81,91):
    
        # Augmentation 9
        
        rot = random.choice(np.arange(-180,179,1))
        if rot ==0:
            rot = 179
        rows,cols,_ = fiximg.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
        dst = cv2.warpAffine(fiximg,M,(cols,rows))
        # write out image
        auglist.append(dst)
        
    elif augshuf in range(91,101):
        
        # Augmentation 10

        rows,cols,ch = fiximg.shape
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(fiximg,M,(cols,rows))
        # write out image
        auglist.append(dst)
    
    return auglist


# ## Function to Load Images
# 
# - The following functions will allow us to take a list of files (saw a batch of files\images) read them using OpenCV, apply augmentation, grab the corresponding target values, and return for training.
# - In training we apply augmentation, but in validation we don't apply augmentation, so we have 2 functions, one for training and one for validation (with no augmentation).

# In[ ]:


def LoadImagesAndTarget_Train(files, lookup_table, shape):
    # initialize variables
    i = 0
    w = shape[0]
    h = shape[1]
    batch_images = np.zeros(((len(files) * 2),w,h,3))
    targetvals = np.zeros(((len(files) * 2),5))
    for file in files:
        # read in image
        img = cv2.imread('../input/train_images/' + file)
        # resize
        img = cv2.resize(img, shape)
        # apply augmentation
        newimages = MyAug(img, shape)
        # normalize
        newimages = np.array(newimages) / 255.0
        # 'newimages' should have our original image and an augmented version
        # so we're creating a new matrix with our original images and augmentations
        # and we include our target value for each
        for img in newimages:
            # add image to batch set
            batch_images[i] = img
            # get the filename without extension
            filename = os.path.splitext(file)[0]
            # get the id from the filename
            id_from_filename = filename[0:filename.find('-')]
            # only keep the row from the lookup table that matches our id
            getrow = lookup_table.loc[lookup_table['PetID'] == id_from_filename]
            # change the format to one-hot encoded, and save to target dataset
            targetvals[i] = np.asmatrix(getrow[[0,1,2,3,4]].values[0], dtype=np.float32)
            # iterate i
            i += 1
    return batch_images, targetvals


# In[ ]:


def LoadImagesAndTarget_Test(files, lookup_table, shape):
    # initialize variables
    i = 0
    w = shape[0]
    h = shape[1]
    batch_images = np.zeros(((len(files)),w,h,3))
    targetvals = np.zeros(((len(files)),5))
    for file in files:
        # read in image
        img = cv2.imread('../input/train_images/' + file)
        # resize
        img = cv2.resize(img, shape)
        # in validation we don't apply augmentation
        # normalize
        img = np.array(img) / 255.0
        batch_images[i] = img
        # get the filename without extension
        filename = os.path.splitext(file)[0]
        # get the id from the filename
        id_from_filename = filename[0:filename.find('-')]
        # only keep the row from the lookup table that matches our id
        getrow = lookup_table.loc[lookup_table['PetID'] == id_from_filename]
        # change the format to one-hot encoded, and save to target dataset
        targetvals[i] = np.asmatrix(getrow[[0,1,2,3,4]].values[0], dtype=np.float32)
        # get target based on filename
        i += 1
    #print("returning batches ...")
    return batch_images, targetvals


# ## Create Image Loader
# 
# - We're using train_on_batch here because it gives us a bit more control and in this case had a better experience over trying to set up a generator.
# - This is pretty basic, we just have to build a lot from scratch since we're using train_on_batch
# - We set up epochs, batches, loading batches, running train_on_batch, printing feedback, storing results, etc.

# In[ ]:


def KerasModelTrainer(files, batch_size, lookup_table, epochs, test_size, shape):
    
    # initialize variables for storing history and calculating batches
    L = len(files)
    rnds = L // batch_size
    training_loss_history_e = []
    test_loss_history_e = []
    training_acc_history_e = []
    test_acc_history_e = []

    for epoch in range(1,epochs + 1):
        
        # initialize variables for storing history and calculating batch ranges
        batch_start = 0
        batch_end = batch_size
        test_cases = int(batch_size * test_size)
        training_loss_history = []
        test_loss_history = []
        training_acc_history = []
        test_acc_history = []
        mycnt = 0
        start = time.time()
        
        print("Epoch {}/{}".format(epoch,epochs))
        
        while batch_start < L:
            
            # initialize variables for calculating batch ranges and printing results
            mycnt += 1
            pct = int((mycnt / rnds) * 100)
            limit = min(batch_end, L)

            # load train and test images for training with augmentation
            Xtrain, Ytrain = LoadImagesAndTarget_Train(files[batch_start:(limit - test_cases)], lookup_table, shape)
            Xtest, Ytest = LoadImagesAndTarget_Test(files[((limit - test_cases)):limit], lookup_table, shape)    

            # train
            model.train_on_batch(Xtrain,Ytrain)

            # test on train
            training_metrics = model.test_on_batch(Xtrain,Ytrain)
            training_loss = training_metrics[0]
            training_acc = training_metrics[1]
            
            # save model training metrics
            training_loss_history.append(training_loss)
            training_acc_history.append(training_acc)
            
            # test on test
            test_metrics = model.test_on_batch(Xtest,Ytest)
            test_loss = test_metrics[0]
            test_acc = test_metrics[1]
            
            # save model test\validation metrics
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)
            
            # update batch window
            batch_start += batch_size   
            batch_end += batch_size
            
            # update overall performance
            if np.isnan(np.mean(training_acc_history)) == False:
                train_acc_mean = np.mean(training_acc_history)
            if np.isnan(np.mean(test_acc_history)) == False:
                test_acc_mean = np.mean(test_acc_history)
            
            # communicate training results so far
            print("Training {}% [".format(pct), 
                  "#" * int(pct/5), 
                  "." * (20 - int(pct/5)), "]", 
                  " Train Acc: {0:.3f} | ".format(train_acc_mean), 
                  " Test Acc: {0:.3f}".format(test_acc_mean), end='\r')
        
        end = time.time()
        print("")
        print("Processing Time: {0:.2f} min".format((end - start) / 60))
        
        # storing training history
        test_loss_history_e.append(np.mean(test_loss_history))
        test_acc_history_e.append(np.mean(test_acc_history))
        training_loss_history_e.append(np.mean(training_loss_history))
        training_acc_history_e.append(np.mean(training_acc_history))
    
    return model, training_loss_history_e, test_loss_history_e, training_acc_history_e, test_acc_history_e


# ## Define Model

# In[ ]:


#create model
model = Sequential()
#add model layers
model.add(conv_base)
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(40, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(30, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(20, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(10, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))


# In[ ]:


conv_base.trainable = False


# In[ ]:


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# ## Run Model

# In[ ]:


# Get list of files
files = os.listdir('../input/train_images') 


# In[ ]:


epochs = 5
model, train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = KerasModelTrainer(files=files,
                                                                                          batch_size=100,
                                                                                          lookup_table=pet_ids_and_target, 
                                                                                          epochs=epochs, 
                                                                                          test_size=0.1, 
                                                                                          shape=(224,224))


# # Training Plots

# In[ ]:


plt.plot(np.arange(1, epochs+1, 1), test_loss_hist)
plt.plot(np.arange(1, epochs+1, 1), train_loss_hist)
plt.xlabel('Rounds / Batches')
plt.ylabel('Loss')
plt.title('Train / Test Loss History')
plt.legend(['test', 'train'], loc='upper left')
plt.grid(True)
plt.show()


# In[ ]:


plt.plot(np.arange(1, epochs+1, 1), test_acc_hist)
plt.plot(np.arange(1, epochs+1, 1), train_acc_hist)
plt.xlabel('Rounds / Batches')
plt.ylabel('Accuracy')
plt.title('Train / Test Acc History')
plt.legend(['test', 'train'], loc='upper left')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




