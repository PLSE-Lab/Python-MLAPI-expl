#!/usr/bin/env python
# coding: utf-8

# ### Pipeline for Stain Normilized Training (Keras)
# With this kernel you can train a Deep Neural Network with stain normalized images. The basic idea is to convert train and test images to the similar color space. All source images are transformed on the basis of a randomly selected target image. During the transformation, however, I had problems with the some images leading to a singular matrix during the conversion. I found these images in the train and test set. Here are some papers describing Stain Normalization:
# 
# - [Neural Stain Normalization and Unsupervised Classification of Cell Nuclei in Histopathological Breast Cancer Images](http://https://arxiv.org/abs/1811.03815)
# - [The importance of stain normalization in colorectal tissue classification with convolutional networks](http://https://arxiv.org/abs/1702.05931)
# - [Stain normalization of histopathology images using generative adversarial networks](http://https://ieeexplore.ieee.org/document/8363641)
# 
# I took and changed the main functions from https://github.com/Peter554/StainTools
# 

# In[ ]:


from sklearn.utils import shuffle
import pandas as pd
import os

# Train 
df = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")

# Test
df_test = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

# Test cleaned
df_test_cleaned = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

df = shuffle(df,random_state=123)

print(len(df))
print(len(df_test))
print(len(df_test_cleaned))


# In[ ]:


# Remove error mages from the training set

df_train_error1 = pd.read_csv('../input/train-error-images1/train_error_images1.csv')
df_train_error2 = pd.read_csv('../input/train-error-images2/train_error_images2.csv')
df_train_error3 = pd.read_csv('../input/train-error-images3/train_error_images3.csv')
df_train_error3 = df_train_error3.drop(df_train_error3.columns[0], axis=1)
df_train_error1.columns = ['id','label']
df_train_error2.columns = ['id','label']
df_train_error3.columns = ['id','label']
df_train_error = pd.concat([df_train_error1,df_train_error1,df_train_error1])
df_train_error = df_train_error.drop_duplicates(subset='id', keep='first')
for i in range(len(df_train_error)):
     df = df[df['id'] != df_train_error.iloc[i,0]]


# In[ ]:


# Remove error images from the test set

df_test_error = pd.read_csv('../input/test-error-images1/test_error_images.csv')
df_test_error = df_test_error.drop(df_test_error.columns[0], axis=1)
df_test_error = df_test_error.drop_duplicates(subset='id', keep='first')

for error in df_test_error['id']:
     df_test_cleaned = df_test_cleaned[df_test_cleaned['id'] != error]


# In[ ]:


# Reduce train set for demonstration only

df = df[0:5000]


# In[ ]:


# Split data set  to train and validation sets

from sklearn.model_selection import train_test_split

# Use stratify= df['label'] to get balance ratio 1/1 in train and validation sets
df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'], random_state=123)

# Check balancing
print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))


# In[ ]:


# Train List
train_list = df_train['id'].tolist()
train_list = ['../input/histopathologic-cancer-detection/train/'+ name + ".tif" for name in train_list]

# Validation List
val_list = df_val['id'].tolist()
val_list = ['../input/histopathologic-cancer-detection/train/'+ name + ".tif" for name in val_list]

# Test list
test_list = df_test['id'].tolist()
test_list = ['../input/histopathologic-cancer-detection/test/'+ name + ".tif" for name in test_list]

# Test cleaned
test_cleaned_list = df_test_cleaned['id'].tolist()
test_cleaned_list = ['../input/histopathologic-cancer-detection/test/'+ name + ".tif" for name in test_cleaned_list]

# Test error
test_error_list = df_test_error['id'].tolist()
test_error_list = ['../input/histopathologic-cancer-detection/test/'+ name + ".tif" for name in test_error_list]

# id dictionary
id_label_map = {k:v for k,v in zip(df.id.values, df.label.values)}


# - Demonstration of images where I could not perform stain normalization because they lead to a singular matrix during the conversion.

# In[ ]:


# Example of images where you will have trouble with stain normalization

import cv2 as cv
import matplotlib.pyplot as plt

fig = plt.figure()
fig, ax = plt.subplots(1,3, figsize=(20,20))

ax[0].imshow(cv.imread(test_error_list[10]))
ax[0].set_title("Image from the Test Set",fontsize=14)
ax[1].imshow(cv.imread(test_error_list[15]))
ax[1].set_title("Image from the Test Set",fontsize=14)
ax[2].imshow(cv.imread(test_error_list[16]))
ax[2].set_title("Image from the Test Set",fontsize=14)


# In[ ]:


# Functions for generators

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np 
import cv2

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[ ]:


# Augmentation

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np 
import cv2

def augmentation():
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    seq = iaa.Sequential([
        # Horizontal flips 50% images  
        iaa.Fliplr(0.5),   
        
        # Hrizontal flips 50% images
        iaa.Flipud(0.5),
        
        # Crop some of the images by 0-10% of their height/width
        #sometimes(iaa.Crop(percent=(0, 0.1))),
        
        #iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))), # gaussian blur
        
        #iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))) # gaussian noise
        
        # Translate images by -20 to +20% on x- and y-axis independently:
        #iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255)),
        
        # Rotate images by -45 to 45 degrees:
        #iaa.Affine(rotate=(-45, 45)),
        
         ], random_order=True) # apply augmenters in random order
    
    return seq


# In[ ]:


get_ipython().system('pip install spams')


# In[ ]:


# STAIN NORMALIZATION FUNCTIONS

import spams

class TissueMaskException(Exception):
    pass

######################################################################################################

def is_uint8_image(I):
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True
######################################################################################################

def is_image(I):
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True
######################################################################################################

def get_tissue_mask(I, luminosity_threshold=0.8):
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")

    return mask

######################################################################################################

def convert_RGB_to_OD(I):
    mask = (I == 0)
    I[mask] = 1
    

    #return np.maximum(-1 * np.log(I / 255), 1e-6)
    return np.maximum(-1 * np.log(I / 255), np.zeros(I.shape) + 0.1)

######################################################################################################

def convert_OD_to_RGB(OD):
    
    assert OD.min() >= 0, "Negative optical density."
    
    OD = np.maximum(OD, 1e-6)
    
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

######################################################################################################

def normalize_matrix_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]

######################################################################################################


def get_concentrations(I, stain_matrix, regularizer=0.01):
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T

######################################################################################################

def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):

    # Convert to OD and ignore background
    tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    
    OD = OD[tissue_mask]

    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

    # The two principle eigenvectors
    V = V[:, [2, 1]]

    # Make sure vectors are pointing the right way
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])

    # Min and max angles
    minPhi = np.percentile(phi, 100 - angular_percentile)
    maxPhi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Order of H and E.
    # H first row.
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    return normalize_matrix_rows(HE)

######################################################################################################

def mapping(target,source):
    
    stain_matrix_target = get_stain_matrix(target)
    target_concentrations = get_concentrations(target,stain_matrix_target)
    maxC_target = np.percentile(target_concentrations, 99, axis=0).reshape((1, 2))
    stain_matrix_target_RGB = convert_OD_to_RGB(stain_matrix_target) 
    
    stain_matrix_source = get_stain_matrix(source)
    source_concentrations = get_concentrations(source, stain_matrix_source)
    maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    source_concentrations *= (maxC_target / maxC_source)
    tmp = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))
    return tmp.reshape(source.shape).astype(np.uint8)


# In[ ]:


# Show example stain transformation
import cv2 as cv
import matplotlib.pyplot as plt

target = cv.imread(test_list[10])
source = cv.imread(train_list[555])

# Convert from cv2 standard of BGR to our convention of RGB.
target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
source = cv.cvtColor(source, cv.COLOR_BGR2RGB)

# Perform stain normalization
transformed = mapping(target,source)

fig = plt.figure()
fig, ax = plt.subplots(1,3, figsize=(20,20))

ax[0].imshow(source)
ax[0].set_title("Source Image",fontsize=14)
ax[1].imshow(target)
ax[1].set_title("Target Image",fontsize=14)
ax[2].imshow(transformed)
ax[2].set_title("Transformed Image",fontsize=14)


# In[ ]:


# Import Pretrained Models
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.models import Sequential
from keras import applications
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate

# Load pretrained model
inputTensor = Input((96,96,3))
model_DenseNet169 = DenseNet169(include_top=False, weights='imagenet')

# Concatenate Pretrained Models
######
#models = [model_NASNet,model_DenseNet201]
models = [model_DenseNet169]
######

outputTensors = [m(inputTensor) for m in models]
if len(models) > 1:
    output = Concatenate()(outputTensors) 
else:
    output = outputTensors[0]
    
# Classifier 
out1 = GlobalMaxPooling2D()(output)
out2 = GlobalAveragePooling2D()(output)
out = Concatenate(axis=-1)([out1, out2])
out = Dropout(0.8)(out)
out = Dense(1, activation="sigmoid")(out)
model = Model(inputTensor,out)
model.summary()


# In[ ]:


# Read and convert images to rgb
import cv2 as cv
import os

def read_image(path):
    im = cv.imread(path)
    # Convert from cv2 standard of BGR to our convention of RGB.
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


# In[ ]:


# Stain Normalization on the fly

from random import randint
import random

import cv2 as cv


def stain_normalization(sources,target_list = test_list):
    
    X = []
    # Chose any random target iamge for stain normalization
    target = cv.imread(train_list[random.randint(0,len(train_list))])
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
    
    for source in sources:     
        
        # Perform stain normalization
        transformed = mapping(target,source)
        
        X.append(transformed)
        
    return np.asarray(X)


# In[ ]:


# Train Generator

def train_gen(list_files, id_label_map, batch_size, augment = False, stain = False):
    
    while True:
        
        shuffle(list_files)
        
        for batch in chunker(list_files, batch_size):
            
            X = [read_image(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]      
            
            if stain:
                X = stain_normalization(X)
            
            if augment:
                X = augmentation().augment_images(X)   
                
            X = [preprocess_input(x) for x in X]   
            yield np.array(X), np.array(Y)
   


# In[ ]:


# Train the model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

batch_size = 64
stain = True


model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001),metrics=['accuracy'])

callbacks =  ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min',verbose=1)

history = model.fit_generator(train_gen(train_list, id_label_map, batch_size, augment = False, stain = stain),
                              validation_data=train_gen(val_list, id_label_map, batch_size, augment = False, stain = stain),
                              epochs = 10,
                              steps_per_epoch = len(train_list) // batch_size + 1,
                              validation_steps = len(val_list) // batch_size + 1,
                              callbacks=[callbacks],
                              verbose=1)


# In[ ]:


# Plot validation and accuracies over epochs
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(range(len(val_loss)),val_loss,'c',label='Validation loss')
plt.plot(range(len(loss)),loss,'m',label='Train loss')

plt.title('Training and validation losses')
plt.legend()
plt.xlabel('epochs')
plt.show()


# In[ ]:


# Predict cleaned stain normalized test data
model.load_weights('model.h5')

preds = []
ids = []

i = len(test_cleaned_list)
for batch in chunker(test_cleaned_list[:10], batch_size):
    X = [read_image(x) for x in batch]
    X = stain_normalization(X)
    X = [preprocess_input(x) for x in X]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = model.predict(X).ravel().tolist()
    preds += preds_batch
    ids += ids_batch
    print(i)
    i = i - 32
    
df_subm_cleaned = pd.DataFrame({'id':ids, 'label':preds})


# In[ ]:


# Predict error test data without stain normalization

preds = []
ids = []

for batch in chunker(test_error_list, batch_size):
    X = [read_image(x) for x in batch]
    X = [preprocess_input(x) for x in X]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = model.predict(X).ravel().tolist()
    preds += preds_batch
    ids += ids_batch

df_subm_error = pd.DataFrame({'id':ids, 'label':preds})


# In[ ]:


df_subm = pd.concat([df_subm_cleaned,df_subm_error])
df_subm.to_csv("submission.csv", index=False)


# In[ ]:


df_subm.head()


# In[ ]:




