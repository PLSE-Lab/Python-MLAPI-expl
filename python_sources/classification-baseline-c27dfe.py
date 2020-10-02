#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# ### Note
# Some methods and codes in this kernel are adapted from another of my kernel, which contains more comprehensive citations: https://www.kaggle.com/tommzzhou/eda-and-preprocessing
# 
# 
# ### Thanks to
# Somshubra Majumdar for his keras implementation of efficient Net
# 
# @ Neuron Engineer for the dataset containing efficient net weights
# 
# @ averagemn for the resized version of the previous dataset
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2
from functools import partial
import scipy as sp


# For transfer learning
import tensorflow_hub
from tensorflow.keras import applications

import matplotlib.pyplot as plt
import seaborn as sns
import random
import albumentations

from tqdm import tqdm
from math import ceil
import math
import sys
import os
import gc

from tensorflow.keras.activations import elu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:4]:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


# Record the starting time of this commit
import time
t_start = time.time()


# In[ ]:


get_ipython().system('ls ../input/efficient-net-keras/efficientnet-master/')


# In[ ]:


# Repository source: https://github.com/qubvel/efficientnet
sys.path.append(os.path.abspath('../input/efficient-net-keras/efficientnet-master/'))

from efficientnet.tfkeras import EfficientNetB5


# In[ ]:


get_ipython().system('which python3')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


# ## Basic Ideas
# 
# * Write things in modules, more of an OOP style
# * Some **well-designed** train time data augmentation
# * Using keras Efficient Net
# * Strong and usable data generator
# 
# 
# To-do List:
# 
# [ ] Fine-tune the network with guidance from CS231n
# 
# [ ] Making use of additional data
# 
# [ ] MixUp augmentations 
# 
# [ ] Ways to deal with imbalanced data?
# 
# [ ] Apply TTA and see if the results are good
# 
# [ ] Ensemble models with Green Channels and Ben's method
# 
# [ ] Clever way to split train and validation data
# 
# 
# 

# ## Preparation
# 
# * Load the files in
# * Check compatability and versions
# * **Set gloabl variables**

# In[ ]:


train_file_name = "/kaggle/input/aptos2019-blindness-detection/train.csv"
test_file_name = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
previous_contest_file_name = "/kaggle/input/retinopathy-train-2015/rescaled_train_896/trainLabels.csv"

preivous_contest_file_path = "/kaggle/input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/"
train_file_path = "/kaggle/input/aptos2019-blindness-detection/train_images/"
test_file_path = "/kaggle/input/aptos2019-blindness-detection/test_images/"


# In[ ]:


dataset_csv = pd.read_csv(train_file_name)
test_csv = pd.read_csv(test_file_name)
previous_contest_csv = pd.read_csv(previous_contest_file_name)


# In[ ]:


print(tf.__version__)
print(cv2.__version__)


# In[ ]:


CLASS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
SEED = 24

BATCHSIZE = 16

IMG_CHANNELS = 3
IMG_WIDTH = 512

# These are used for histogram equalization
clipLimit=2.0 
tileGridSize=(8, 8)  


SAVED_MODEL_NAME = "efficientB5.hdf5"
CUSTOM_WEIGHT = "/kaggle/input/efficientn5/efficientB5.hdf5"

channels = {"R":0, "G": 1, "B":2}


# In[ ]:


dataset_csv.head(5)


# In[ ]:


test_csv.head(5)


# In[ ]:


previous_contest_csv.columns = ["id_code", "diagnosis"]


# In[ ]:


previous_contest_csv.head(5)


# ## Split and Check Data

# In[ ]:


# Check the random seed
print(SEED)


# #### Split the data
# Using stratify param ensures that the training and validation data have the same label distribution

# In[ ]:


x_training, x_validation, y_training, y_validation = train_test_split(dataset_csv["id_code"], dataset_csv["diagnosis"],
                                                    test_size=0.15,
                                                    random_state=SEED,
                                                    stratify=dataset_csv["diagnosis"])


# In[ ]:


print(type(x_training))
print(x_training[:5])
print(type(x_validation))
print(x_validation[:5])

# Now check for y labels
print(type(y_validation))
print(y_validation[:5])


# In[ ]:


# Merge the x_training and y_training to a single dataframe 
# This is done for future convenience

train_csv = pd.DataFrame(columns = ['id_code', 'diagnosis'])
train_csv["id_code"] = x_training
train_csv["diagnosis"] = y_training

valid_csv = pd.DataFrame(columns = ['id_code', 'diagnosis'])
valid_csv["id_code"] = x_validation
valid_csv["diagnosis"] = y_validation

# Should re-index these newly merged dataframes
train_csv.reset_index(inplace = True)
valid_csv.reset_index(inplace = True)


# In[ ]:


print(len(dataset_csv["diagnosis"]))
print(len(train_csv["diagnosis"]))
print(len(valid_csv["diagnosis"]))


# #### Now check the distributions

# In[ ]:


dataset_csv["diagnosis"].hist(figsize = (8,4))


# In[ ]:


train_csv['diagnosis'].hist(figsize = (8,4))


# In[ ]:


valid_csv['diagnosis'].hist(figsize = (8,4))


# #### Display some images from the dataset folder

# In[ ]:


def display_samples(df, columns=4, rows=3, img_dir = train_file_path):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    
    random.seed(SEED)
    random_indices = random.sample(range(0, len(df)), columns*rows)
    count = 0
    
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        
        full_path = os.path.join(img_dir, image_path+".png")
        
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()

# display_samples(train_csv, 6, 2)


# In[ ]:


# display_samples(previous_contest_csv, 6, 2, img_dir = "/kaggle/input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/")


# ## Check Pre-processing of Data

# In[ ]:


def crop_image(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
        return img


# ### Single Channel Method

# In[ ]:


def display_single_channel_samples_crop_resize(df, columns=4, rows=3, channel = "G", HE = False, img_dir = train_file_path):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    random.seed(SEED) # This lines make sure that all the following function calls will
                    # show the same set of randomly selected images
    random_indices = random.sample(range(0, len(df)), columns*rows)
    count = 0
    
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        
        full_path = os.path.join(img_dir, image_path+".png")
        
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = crop_image(img)
        img = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))
        # Apply some pre-processing
        img = img[:,:,channels[channel]]
        if HE: #If the histogram equalization is applied
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            img = clahe.apply(img) #This is for creating the image with a higher contrast
        else:
            pass
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()
    


# In[ ]:


# display_single_channel_samples_crop_resize(train_csv, 6,2, HE = True)


# ### Ben's method

# In[ ]:



def display_samples_bens_crop_resize(df, columns=4, rows=3, sigmaX = 15, img_dir = train_file_path):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    random.seed(SEED) # This lines make sure that all the following function calls will
                    # show the same set of randomly selected images
    random_indices = random.sample(range(0, len(df)), columns*rows)
    count = 0
    
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        
        full_path = os.path.join(img_dir, image_path+".png")
        
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = crop_image(img)
        img = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))
        
        # This following line is the key to ben's method
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()


# In[ ]:


# display_samples_bens_crop_resize(train_csv, 6, 2)


# ## Image Generator with Augmentations

# ### Finalize Image Pre-processor
# 
# In the above cases, we used two different pre-processing methods and compared their results. Here we finalize these two into a generator function. Given the image directly retrieved from the dataset, the generator produces, based on user's choice, pre-processed image. 
# 
# 
# For the pre-processor to work, the image feed in must be in RGB. Note that this function does ** cropping and resizing ** by itself.

# In[ ]:


# Check the values in this dictionary
print(channels)
print(clipLimit)
print(tileGridSize)

def pre_process(img, method = "SingleChannel", channel = "G", sigmaX = 15):
    
    img = crop_image(img)
    img = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))
    
    
    if method == "Bens":
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)
#         print("Bens")
    elif method == "SingleChannel":
        single_ch = img[:,:,channels[channel]]
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        single_ch = clahe.apply(single_ch)
        img = cv2.merge([single_ch, single_ch, single_ch])
#         print("Single Channel")
    else:
        print("Argument Error: Can't recognize METHOD passed in")
        
    return img


# ### Augmentations
# 
# We used albumentations provided by the albumentations library. Stronly recommended.
# 
# * Random Brightness
# * Rotation
# * Median Blur
# * Gaussian Noise
# * Horizontal and vertical flip
# * Random Gamma and Alpha
# * Hue and Saturation
# 
# 

# In[ ]:


from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, VerticalFlip,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p = 0.9),
    VerticalFlip(p = 0.9),
    RandomContrast(limit = 0.2, p = 0.8), # to do

    OneOf([
        RandomGamma(gamma_limit = (90,100), p = 0.5),
        MedianBlur(blur_limit = 3, p = 0.5)
    ], p = 0.6),
    
    OneOf([
        ShiftScaleRotate(rotate_limit=25, p = 0.8),
        OpticalDistortion(p = 0.4)
    ], p = 1),

    OneOf([
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        RandomBrightness(),
    ], p=0.8)
    
],p=1)


AUGMENTATIONS_TEST = Compose([
#     ToFloat(max_value=1)
],p=1)


# #### Experiment with these image augmentation methods
# 
# The following function will take a few random images from the dataset provided and then do heavy augmentation repeatedly on them. Afterwards, this function will show the results.

# In[ ]:



def test_augmentations(df, num, aug_num, augment, img_dir = train_file_path):
    # For num images, perform aug_num times of augmentations on each
    
    total_num = num*aug_num
    fig=plt.figure(figsize=(6*aug_num, 4*num))
    
    # Randomly select num images
    random_indices = random.sample(range(0, len(df)), num)
    count = 0
    
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        full_path = os.path.join(img_dir, image_path+".png")
        
        # Read in the image and transform into RGB
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pre_process(img, method = "SingleChannel")
        
        for j in range(aug_num):
            augmented = augment(image = img)
            fig.add_subplot(num, aug_num, count+1)
            plt.title(image_rating)
            plt.imshow(augmented["image"])
            count += 1
    
    plt.tight_layout()


# In[ ]:


test_augmentations(train_csv, 4, 6, augment = AUGMENTATIONS_TRAIN)


# In[ ]:


print(BATCHSIZE)
print(train_file_path)
class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
    def __init__(self,
                 df = train_csv,
                 augmentation=None, batch_size=BATCHSIZE,
                 img_size=IMG_WIDTH, n_channels=IMG_CHANNELS, 
                 shuffle=True,
                image_dir = train_file_path):

        self.dataframe = df
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentation
        
        # Get all the paths out from the specified dataframe
        self.img_paths = [filename for filename in self.dataframe["id_code"]]
#         print("self.img_paths: ", self.img_paths[:10])
        self.img_dir = image_dir
        
        self.on_epoch_end()
        
    def __len__(self):
#         'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.img_paths))]

        # Select the next set of img paths to load the corresponding images and labels
        list_IDs_im = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)
        
        # First generate data
        # Then pre-process and augment them
        return X.astype(np.float32)/255, y
    
    def pre_process_augment(self, img):
        # Use the written pre-process function here
        img = pre_process(img, method = "SingleChannel")
        
        if self.augment == None:
            return img
        else: 
            augmented = self.augment(image = img)
            aug_im = augmented['image']
    #       print("augmented image data type:", aug_im.dtype)
            return aug_im

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im),self.img_size,self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), 1))
        
        # Generate data
        for i, im_path in enumerate(list_IDs_im):
            full_path = os.path.join(self.img_dir, im_path+".png")
#             print(full_path)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Perform image augmentation here
            aug_im = self.pre_process_augment(img)
            
            X[i,] = aug_im
            y[i] = self.dataframe.loc[self.dataframe["id_code"] == im_path, "diagnosis"]
            
        return X,y 


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'a = DataGenerator(batch_size=6,shuffle=False, augmentation=AUGMENTATIONS_TRAIN)\ngen_images,gen_labels = a.__getitem__(0)')


# ## Model Architecture

# ### Instantiate Data Generators

# In[ ]:


# The train images and validation images come from the same folder, no need to specify
train_generator = DataGenerator(df = train_csv, batch_size = 6, shuffle = True, augmentation = AUGMENTATIONS_TRAIN)
valid_generator = DataGenerator(df = valid_csv, batch_size = 6, shuffle = False, augmentation = None)

previous_generator = DataGenerator(df = previous_contest_csv, batch_size = 6, shuffle = True, augmentation = None, image_dir = preivous_contest_file_path)


# In[ ]:


gen_images,gen_labels = train_generator.__getitem__(0)

for i in range(2):
    img = gen_images[i]
    plt.imshow(img)
    plt.title(gen_labels[i])
    plt.show()


# In[ ]:


get_ipython().system('ls /kaggle/input/retinopathy-train-2015/rescaled_train_896')


# In[ ]:


gen_images,gen_labels = previous_generator.__getitem__(0)

for i in range(2):
    img = gen_images[i]
    plt.imshow(img)
    plt.title(gen_labels[i])
    plt.show()


# In[ ]:


gen_images,gen_labels = valid_generator.__getitem__(0)

for i in range(2):
    img = gen_images[i]
    plt.imshow(img)
    plt.title(gen_labels[i])
    plt.show()


# ### Metrics, Custom Loss, and Checkpointing

# In[ ]:


print(valid_generator.__len__())
print(train_generator.__len__())
print(previous_generator.__len__())


# Cite the below code:

# In[ ]:


def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    
    preds = []
    labels = []
    
    for index in range(generator.__len__()):
        x, y = generator.__getitem__(index)
        preds.append(model.predict(x))
        labels.append(y)
    
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()


# In[ ]:


class CustomMetrics(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """
    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, valid_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        
        # We can use sklearns implementation of QWK straight out of the box
        # as long as we specify weights as 'quadratic'
        
        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(SAVED_MODEL_NAME)
        return


# 
# ### Simple Custom Model
# 
# This part has not been completed yet

# 
# 
# ### Transfer Learning

# #### Examine the loaded pre-trained Efficient Net
# 
# We can see that the last three layers of the with_top model are removed. The three layers are : avg_pool, top_dropout, probs
# 

# In[ ]:


# # Load in EfficientNetB5
# effnet = None
# effnet = EfficientNetB5(weights=None,
#                         include_top=False,
#                         input_shape=(IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS))
# effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')


# In[ ]:


# effnet.summary()


# #### Add some structures

# Citation to this code

# In[ ]:


# layer = effnet.layers[0]
# print(type(layer))
# print(layer.trainable)
# print(len(effnet.layers))


# #### Freeze some layers
# 
# This is just a trial, see what score we can get

# In[ ]:


# for layer in effnet.layers[:100]:
#     layer.trainable = False


# In[ ]:


# # Check the freezing is successful
# layer = effnet.layers[0]
# print(type(layer))
# print(layer.trainable)


# In[ ]:


# model = keras.Sequential()
# model.add(effnet)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.4))
# model.add(Dense(12, activation=elu)) # How to modify this param?
# model.add(Dense(1, activation="linear"))


# model.compile(loss='mse',
#             optimizer=Adam(lr=0.0001), 
#             metrics=['mse', 'acc'])

# print(model.summary())


# #### Instantiate some custom callbacks and train the model

# In[ ]:


# Directly load the model
print(CUSTOM_WEIGHT)
model = load_model(CUSTOM_WEIGHT)


# In[ ]:


kappa_metrics = CustomMetrics()

# Monitor MSE to avoid overfitting and save best model
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)

rlr = ReduceLROnPlateau(monitor='val_loss', 
                        factor=0.6, 
                        patience=6, 
                        verbose=1, 
                        mode='auto', 
                        epsilon=0.0001)

# Begin training
# No augmentations ==> a bit faster

history = model.fit_generator(train_generator,
                    steps_per_epoch = train_generator.__len__(),
                    epochs = 2,
                    workers = 6,
                    validation_data=valid_generator,
                    validation_steps = valid_generator.__len__(),
                    callbacks=[kappa_metrics, es, rlr],
                   verbose = 1)


# ### Download the trained weight

# In[ ]:


# !ls


# In[ ]:


# from IPython.display import FileLinks
# FileLinks('.')


# ### Make predictions with the best performing model

# #### Visualize the training procedure

# In[ ]:


# history_df = pd.DataFrame(model.history.history)
# history_df[['loss', 'val_loss']].plot(figsize=(12,5))
# plt.title("Loss (MSE)", fontsize=16, weight='bold')
# plt.xlabel("Epoch")
# plt.ylabel("Loss (MSE)")
# history_df[['acc', 'val_acc']].plot(figsize=(12,5))
# plt.title("Accuracy", fontsize=16, weight='bold')
# plt.xlabel("Epoch")
# plt.ylabel("% Accuracy");


# In[ ]:


# model.load_weights(SAVED_MODEL_NAME)


# #### See how the best performing model performs

# In[ ]:


# # Calculate QWK on train set
# y_train_preds, train_labels = get_preds_and_labels(model, train_generator)
# y_train_preds = np.rint(y_train_preds).astype(np.uint8).clip(0, 4)

# # Calculate score
# train_score = cohen_kappa_score(train_labels, y_train_preds, weights="quadratic")

# Calculate QWK on validation set
y_val_preds, val_labels = get_preds_and_labels(model, valid_generator)
y_val_preds = np.rint(y_val_preds).astype(np.uint8).clip(0, 4)

# Calculate score
val_score = cohen_kappa_score(val_labels, y_val_preds, weights="quadratic")


# In[ ]:


print(val_score)


# #### Using threshold optimizer

# In[ ]:


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


# Optimize on validation data and evaluate again
y_val_preds, val_labels = get_preds_and_labels(model, valid_generator)
optR = OptimizedRounder()
optR.fit(y_val_preds, val_labels)
coefficients = optR.coefficients()
opt_val_predictions = optR.predict(y_val_preds, coefficients)
new_val_score = cohen_kappa_score(val_labels, opt_val_predictions, weights="quadratic")


# In[ ]:


print("Old validation score:", val_score)
print("New validation score:", new_val_score)
print("New coefficients:", coefficients)


# ## Predict and submit

# In[ ]:


# Notice, must use image data generator
print(test_file_path)
test_generator = DataGenerator(df = test_csv, batch_size = 6, shuffle = False, augmentation = None, image_dir = test_file_path)


# In[ ]:


y_test = model.predict_generator(test_generator)


# In[ ]:


print(y_test.shape)


# In[ ]:


y_test = optR.predict(y_test, coefficients).astype(np.uint8)
test_csv['diagnosis'] = y_test
test_csv.to_csv('submission.csv', index=False)


# Have a basic sense of how much time this kernel takes to train and inference

# In[ ]:


test_csv.head(20)


# In[ ]:


t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)
print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 
                                                      int(total_time*60)))

