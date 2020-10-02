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


# # References:
# 
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

# # Imports

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
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.applications import DenseNet121
from keras import backend as K
from keras.optimizers import Adam


# # View Image Shapes

# In[ ]:


print(cv2.imread('../input/aptos2019-blindness-detection/train_images/000c1434d8d7.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/4289af3afbd2.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/810ed108f5b7.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/be521870a0ea.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/001639a390f0.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/4294a14c656a.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/8114d6a160df.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/be68322c7223.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/0024cdab0c1e.png').shape)
print(cv2.imread('../input/aptos2019-blindness-detection/train_images/42985aa2e32f.png').shape)


# # View an Image

# In[ ]:


# read in image
img = cv2.imread('../input/aptos2019-blindness-detection/train_images/f549294e12e1.png')
#img = cv2.imread('../input/aptos2019-blindness-detection/train_images/000c1434d8d7.png')
img = cv2.resize(img, (224,224))
# show image
plt.imshow(img)


# In[ ]:


np.max(img)


# # Set Up ID & Target Lookup Table

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
ids_and_target = train_df[['id_code','diagnosis']]
onehottarget = pd.get_dummies(ids_and_target['diagnosis'])
ids_and_target = ids_and_target.join(onehottarget)
ids_and_target.iloc[:,5] = np.where(ids_and_target.iloc[:,6]>ids_and_target.iloc[:,5], ids_and_target.iloc[:,6], ids_and_target.iloc[:,5])
ids_and_target.iloc[:,4] = np.where(ids_and_target.iloc[:,5]>ids_and_target.iloc[:,4], ids_and_target.iloc[:,5], ids_and_target.iloc[:,4])
ids_and_target.iloc[:,3] = np.where(ids_and_target.iloc[:,4]>ids_and_target.iloc[:,3], ids_and_target.iloc[:,4], ids_and_target.iloc[:,3])
ids_and_target.iloc[:,2] = np.where(ids_and_target.iloc[:,3]>ids_and_target.iloc[:,2], ids_and_target.iloc[:,3], ids_and_target.iloc[:,2])
print(ids_and_target.head())


# In[ ]:


# test out diagnosis extraction code
getrow = ids_and_target.loc[ids_and_target['id_code'] == 'f549294e12e1']
print(getrow)
print(np.asarray(getrow[[0,1,2,3,4]].values[0], dtype=np.float32))


# In[ ]:


to_categorical(ids_and_target['diagnosis'])


# # My Data Augmentation Function

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
    
    # Augmentations 8-12
        # Augmentation 8
    
    #for i in range(8, 13):
        # generate random percentages
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
    # Augmentations 13-18
    
    #i = 13
    #for rot in [-40, -30, -20, 20, 30, 40]:
        rot = random.choice(np.arange(-180,179,1))
        if rot ==0:
            rot = 179
        rows,cols,_ = fiximg.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
        dst = cv2.warpAffine(fiximg,M,(cols,rows))
        auglist.append(dst)
        #i += 1
        
    elif augshuf in range(91,101):
        
        # Augmentation 19

        rows,cols,ch = fiximg.shape
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(fiximg,M,(cols,rows))
        # write out image
        auglist.append(dst)
    
    return auglist


# # Function to Load Images

# In[ ]:


def LoadImagesAndTarget_Train(files, lookup_table, shape):
    i = 0
    w = shape[0]
    h = shape[1]
    batch_images = np.zeros(((len(files) * 2),w,h,3))
    targetvals = np.zeros(((len(files) * 2),5))
    for file in files:
        # read in image
        img = cv2.imread('../input/aptos2019-blindness-detection/train_images/' + file)
        # resize
        img = cv2.resize(img, shape)
        #print("Starting image augmentation ...")
        newimages = MyAug(img, shape)
        # normalize
        newimages = np.array(newimages) / 255.0
        #print("Loading batch images ...")
        for img in newimages:
            # preprocessing for training, adds 'samples' dim
            #fiximg = preprocess_input(img)
            # add image to training set
            batch_images[i] = img
            # get the filename without extension
            filename = os.path.splitext(file)[0]
            # only keep the row from the lookup table that matches our id
            getrow = lookup_table.loc[lookup_table['id_code'] == filename]
            # change the format to one-hot encoded, and save to target dataset
            targetvals[i] = np.asmatrix(getrow[[0,1,2,3,4]].values[0], dtype=np.float32)
            # get target based on filename
            i += 1
    #print("returning batches ...")
    return batch_images, targetvals


# In[ ]:


def LoadImagesAndTarget_Test(files, lookup_table, shape):
    i = 0
    w = shape[0]
    h = shape[1]
    batch_images = np.zeros(((len(files)),w,h,3))
    targetvals = np.zeros(((len(files)),5))
    for file in files:
        # read in image
        img = cv2.imread('../input/aptos2019-blindness-detection/train_images/' + file)
        # resize
        img = cv2.resize(img, shape)
        #print("Starting image augmentation ...")
        # Note: No augmentation on test data
        # normalize
        img = np.array(img) / 255.0
        batch_images[i] = img
        # get the filename without extension
        filename = os.path.splitext(file)[0]
        # only keep the row from the lookup table that matches our id
        getrow = lookup_table.loc[lookup_table['id_code'] == filename]
        # change the format to one-hot encoded, and save to target dataset
        targetvals[i] = np.asmatrix(getrow[[0,1,2,3,4]].values[0], dtype=np.float32)
        # get target based on filename
        i += 1
    #print("returning batches ...")
    return batch_images, targetvals


# # Create Image Loader

# In[ ]:


def KerasModelTrainer(files, batch_size, lookup_table, epochs, test_size, shape):
    
    # initialize
    L = len(files)
    rnds = L // batch_size
    training_loss_history_e = []
    test_loss_history_e = []
    training_acc_history_e = []
    test_acc_history_e = []
    #this line is just to make the generator infinite, keras needs that    
    #while True:

    for epoch in range(1,epochs + 1):
        
        # initialize
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
            
            # initialize
            mycnt += 1
            pct = int((mycnt / rnds) * 100)
            limit = min(batch_end, L)

            Xtrain, Ytrain = LoadImagesAndTarget_Train(files[batch_start:(limit - test_cases)], lookup_table, shape)
            Xtest, Ytest = LoadImagesAndTarget_Test(files[((limit - test_cases)):limit], lookup_table, shape)    

            # train
            # calculate learning rate
            #current_learning_rate = 0.001 * (10**(-mycnt))
            # train model:
            #K.set_value(model.optimizer.lr, current_learning_rate)  # set new lr
            model.train_on_batch(Xtrain,Ytrain)

            # test on train
            training_metrics = model.test_on_batch(Xtrain,Ytrain)
            training_loss = training_metrics[0]
            training_acc = training_metrics[1]
            #print(training_acc)
            
            training_loss_history.append(training_loss)
            training_acc_history.append(training_acc)
            #print(training_acc_history)
            #print(np.mean(training_acc_history))
            
            # test on test
            test_metrics = model.test_on_batch(Xtest,Ytest)
            test_loss = test_metrics[0]
            test_acc = test_metrics[1]
            
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)
            
            batch_start += batch_size   
            batch_end += batch_size
            
            if np.isnan(np.mean(training_acc_history)) == False:
                train_acc_mean = np.mean(training_acc_history)
            if np.isnan(np.mean(test_acc_history)) == False:
                test_acc_mean = np.mean(test_acc_history)
            
            print("Training {}% [".format(pct), 
                  "#" * int(pct/5), 
                  "." * (20 - int(pct/5)), "]", 
                  " Train Acc: {0:.3f} | ".format(train_acc_mean), 
                  " Test Acc: {0:.3f}".format(test_acc_mean), end='\r')
        
        end = time.time()
        print("")
        print("Processing Time: {0:.2f} min".format((end - start) / 60))
        
        test_loss_history_e.append(np.mean(test_loss_history))
        test_acc_history_e.append(np.mean(test_acc_history))
        training_loss_history_e.append(np.mean(training_loss_history))
        training_acc_history_e.append(np.mean(training_acc_history))
    
    return model, training_loss_history_e, test_loss_history_e, training_acc_history_e, test_acc_history_e


# # Define Model

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenetkeras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


#create model
model = Sequential()
#add model layers
model.add(densenet)
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))


# In[ ]:


#densenet.trainable = False


# In[ ]:


#compile model using accuracy to measure model performance
model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# # Run Model

# In[ ]:


#files = os.listdir('../input/aptos2019-blindness-detection/train_images') 
#files = random.sample(files, 2000)


# In[ ]:


files = os.listdir('../input/aptos2019-blindness-detection/train_images') 


# In[ ]:


epochs = 12
model, train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = KerasModelTrainer(files=files,
                                                                                          batch_size=32,
                                                                                          lookup_table=ids_and_target, 
                                                                                          epochs=epochs, 
                                                                                          test_size=0.15, 
                                                                                          shape=(224,224))


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


# # Score Images

# In[ ]:


files = os.listdir('../input/aptos2019-blindness-detection/test_images')
i = 0
for file in files:
    batch_images = np.zeros((1,224,224,3))
    # read in image
    img = cv2.imread('../input/aptos2019-blindness-detection/test_images/' + file)
    # resize
    img = cv2.resize(img, (224,224))
    # normalize
    img = np.array(img) / 255.0
    #get filename
    filename = os.path.splitext(file)[0]
    # reformat img
    batch_images[0] = img
    # score image
    pred = model.predict(batch_images, verbose=0)
    if i == 0:
        # add prediction and id to a dataframe
        d = {'pred': [pred], 'id_code': [filename]}
        df = pd.DataFrame(data=d)
        i += 1
    else:
        # add prediction and id to a dataframe
        d = {'pred': [pred], 'id_code': [filename]}
        tmp = pd.DataFrame(data=d)
        df = df.append(tmp, ignore_index=True)
        i += 1
        print("Number of images completed: {}".format(i), end='\r')


# In[ ]:


df["diagnosis"] = df["pred"].apply(lambda x: sum(sum((x>0.5).astype(int)))-1)
submission = df[['id_code','diagnosis']]
submission.head()


# In[ ]:


df["diagnosis"].value_counts()


# In[ ]:


submission.to_csv('submission.csv',index=False)

