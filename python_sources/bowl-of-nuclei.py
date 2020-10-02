#!/usr/bin/env python
# coding: utf-8

# This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.
# 
# Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:
# 
# images contains the image file.
# masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# I used code from https://www.kaggle.com/kmader/nuclei-overview-to-submission to load the data
train_labels = pd.read_csv('../input/stage1_train_labels.csv')
train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])
all_images = glob(os.path.join('../input/', 'stage1_*', '*', '*', '*'))
img_df = pd.DataFrame({'path': all_images})
img_id = lambda in_path: in_path.split('/')[-3]
img_type = lambda in_path: in_path.split('/')[-2]
img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
img_df['ImageId'] = img_df['path'].map(img_id)
img_df['ImageType'] = img_df['path'].map(img_type)
img_df['TrainingSplit'] = img_df['path'].map(img_group)
img_df['Stage'] = img_df['path'].map(img_stage)
train_df = img_df.query('TrainingSplit=="train"')
train_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in train_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['masks'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    train_rows += [c_row]
train_img_df = pd.DataFrame(train_rows)    
IMG_CHANNELS = 3
def read_and_stack(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
train_img_df['masks'] = train_img_df['masks'].map(read_and_stack).map(lambda x: x.astype(int))


# In[ ]:


test_df = img_df.query('TrainingSplit=="test"')
test_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in test_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    test_rows += [c_row]
test_img_df = pd.DataFrame(test_rows)    

test_img_df['images'] = test_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])


# Let' check out these nuclei

# In[ ]:


f,axa = plt.subplots(1,2,figsize = (12,5))
axa[0].imshow(train_img_df['images'][2])
axa[1].imshow(train_img_df['images'][5])


# We are dealing with images with different size som RGB some in Grey Scale. I

# In[ ]:


train_img_df['images'].map(lambda x: x.shape).value_counts()


# I will split the dataset 70% for training and 30% for Test/Validation.

# In[ ]:


import math
df = train_img_df.sample(frac=1, random_state= 42)
Train = df[0:math.floor(len(df)*0.7)]
Validation = df[len(Train):]
Test = test_img_df


# Then I define a function to resize all imges and turn them to 1 channel.

# In[ ]:


from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

WIDTH = 128
HEIGHT = 128

def process(data,notest=True):
    X = []
    Y = []
    print("Resizing all...")
    for i in range(len(data.images)):
        img = resize(data.images.iloc[i], (HEIGHT, WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
        X.append(img)
        if(notest):
            img = resize(data.masks.iloc[i], (HEIGHT, WIDTH,1), mode='constant', preserve_range=True)
            Y.append(img)
    print("Done")    
    return X, Y

    print("Turning all to 1 Channel")
    images= []
    for i in range(len(data.images)):
        img = np.zeros([WIDTH,HEIGHT])
        for r in range(len(X[i])):
            for c in range(len(X[i][r])):
                img[r][c] = X[i][r][c].mean()
        img = np.asarray(img).reshape(WIDTH,HEIGHT,1)
        images.append(img)  
    X = images
    if(notest):
        for i in range(len(data.images)):
            Y[i] = np.asarray(Y[i]).reshape(WIDTH,HEIGHT,1)
    print('Done')
    return X, Y
        


# In[ ]:


print("Processing Train")
X_train,Y_train = process(Train)
print("Processing Test")
X_val,Y_val = process(Validation)


# In[ ]:


X_test,Y_test = process(Test,notest= False)


# In[ ]:


np.asarray(X_test[0]).shape


# In[ ]:


n_img = 6
fig, axa = plt.subplots(2, 3, figsize = (15, 6))
axa[0][0].imshow(X_train[0].reshape(WIDTH,HEIGHT,3))
axa[0][1].imshow(X_train[1].reshape(WIDTH,HEIGHT,3))
axa[0][2].imshow(X_train[2].reshape(WIDTH,HEIGHT,3))
axa[1][0].imshow(Y_train[0].reshape(WIDTH,HEIGHT))
axa[1][1].imshow(Y_train[1].reshape(WIDTH,HEIGHT))
axa[1][2].imshow(Y_train[2].reshape(WIDTH,HEIGHT))

plt.show()


# This is the first time I design a U-net, the Idea is well depicted in the following picture (which is taken from the original U Net paper). The idea is the network is composed of two paths: a contracting path (left side of the figure) and an expansive path (on the right side of the figure). The former consists of the repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation for downsampling.  Every step in the expansive path consists of an upsampling of the curren feature map followed by a convolution, a concatenation with the correspondingly 
# feature map from the contracting path, and convolutions again with RELU activation.
# 
# ![](http://tuatini.me/content/images/2017/09/u-net-architecture-1.png)
# 
# 
# here follows some other kernels bout U-nets
# https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
# https://www.kaggle.com/toregil/a-lung-u-net-in-keras
# 

# In[ ]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


# In[ ]:


from keras.models import Model
from keras.layers import *
from keras.layers import UpSampling2D
from keras.callbacks import * 
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


# In[ ]:


input_layer = Input(shape=np.asarray(X_train).shape[1:])
c1 = Conv2D(filters=8,
            input_shape=[WIDTH,HEIGHT,IMG_CHANNELS],
            kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)

c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)

c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)

c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)

#we concatenate the c3 output with the upsample of c4 and apply a further convolution
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)

#the same up to the first layer
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)

l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)

l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)

l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model = Model(input_layer, output_layer)


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.summary()


# In[ ]:


earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(np.asarray(X_train), np.asarray(Y_train), validation_split=0.1, batch_size=16, epochs=10, 
                    callbacks=[earlystopper, checkpointer])


# ## Validation

# Let's see how the trained U-Net performes against the Validation dataset

# In[ ]:


preds = model.predict(np.asarray(X_val))


# In[ ]:


n_img = 6
fig, axa = plt.subplots(2, 3, figsize = (15, 6))
axa[0][0].imshow(X_val[0].reshape(WIDTH,HEIGHT,3))
axa[0][1].imshow(X_val[1].reshape(WIDTH,HEIGHT,3))
axa[0][2].imshow(X_val[2].reshape(WIDTH,HEIGHT,3))
axa[1][0].imshow(preds[0].reshape(WIDTH,HEIGHT))
axa[1][1].imshow(preds[1].reshape(WIDTH,HEIGHT))
axa[1][2].imshow(preds[2].reshape(WIDTH,HEIGHT))


# In[ ]:


from skimage.filters import threshold_otsu
fpreds =[]
for p in preds:
    #thresh = threshold_otsu(p)
    thresh = 0.5
    binary = p > thresh
    fpreds.append(binary)


# In[ ]:


def iou(A,B):
    intersect = (A*B)
    union = (A+B)>0
    return intersect.sum()/union.sum()


# In[ ]:


ious = []
for i in range(len(Y_val)):
    ious.append(iou(Y_val[i],fpreds[i]))


# In[ ]:


pd.Series(ious).mean()


# ## Test

# In[ ]:


tpreds = model.predict(np.asarray(X_test))


# In[ ]:


fpreds = []
for p in tpreds:
    #thresh = threshold_otsu(p)
    thresh = 0.5
    binary = p > thresh
    fpreds.append(binary)


# In[ ]:


fpreds = tpreds


# In[ ]:


fig, axa = plt.subplots(2, 3, figsize = (15, 6))
axa[0][0].imshow(X_test[0].reshape(WIDTH,HEIGHT,3))
axa[0][1].imshow(X_test[10].reshape(WIDTH,HEIGHT,3))
axa[0][2].imshow(X_test[2].reshape(WIDTH,HEIGHT,3))
axa[1][0].imshow(tpreds[0].reshape(WIDTH,HEIGHT))
axa[1][1].imshow(tpreds[10].reshape(WIDTH,HEIGHT))
axa[1][2].imshow(tpreds[2].reshape(WIDTH,HEIGHT))


# In[ ]:


test_sizes = Test.images.map(lambda x: x.shape)


# In[ ]:


preds_test_upsampled = []
for i in range(len(tpreds)):
    preds_test_upsampled.append(resize(np.squeeze(tpreds[i]), 
                                       (test_sizes[i][0], test_sizes[i][1]), 
                                       mode='constant', preserve_range=True))


# In[ ]:


fig, axa = plt.subplots(1, 3, figsize = (15, 6))
axa[0].imshow(preds_test_upsampled[0].reshape(test_sizes[0][0], test_sizes[0][1]))
axa[1].imshow(preds_test_upsampled[1].reshape(test_sizes[1][0], test_sizes[1][1]))
axa[2].imshow(preds_test_upsampled[2].reshape(test_sizes[2][0], test_sizes[2][1]))


# Thanks to https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

# In[ ]:


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# In[ ]:


new_test_ids = []
rles = []
for n, id_ in enumerate(Test.ImageId):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))


# In[ ]:


sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)


# In[ ]:




