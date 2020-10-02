#!/usr/bin/env python
# coding: utf-8

# Another interesting competiton. Let's deep dive into the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import cv2
import glob
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
import imgaug as aug
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Conv2DTranspose, UpSampling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications import resnet50
from keras.applications import densenet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.preprocessing import MultiLabelBinarizer
import imgaug as ia
from imgaug import augmenters as iaa
from keras import backend as K
import tensorflow as tf
from collections import defaultdict, Counter
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


seed=1234

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(seed)

# Set the random seed in tensorflow at graph level
tf.set_random_seed(seed)

# Make the augmentation sequence deterministic
aug.seed(seed)

color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Define some paths for future use
input_path = Path("../input/human-protein-atlas-image-classification")
train_dir = input_path / "train"
test_dir = input_path / "test"
train_csv = input_path / "train.csv"
submission_file = input_path / "sample_submission.csv"


# In[ ]:


# Load keras weights into keras cache folder
# Check for the directory and if it doesn't exist, make one.
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
    
# make the models sub-directory
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# Copy the weights from your input files to the cache directory
get_ipython().system('cp ../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/')


# In[ ]:


# Load the training CSV as it contains all the information
train_df = pd.read_csv(str(train_csv))
train_df.head()


# How many samples are there in the training set? More importatntly, how may unqiue samples are present?

# In[ ]:


print(f"Total number of samples in the training data: {train_df.shape[0]}")
print(f"Total number of unique IDs in the training data: {len(train_df['Id'].unique())}")


# For each image, there can be multiple labels as it is a multiclass-multilabel classification problem. We are going to create another column
# in our dataframe which tracks the number of labels associated with each image

# In[ ]:


# Lets' split up the target column and see how many files containing multiple labels
train_df["nb_labels"] = train_df["Target"].apply(lambda x: len(x.split(" ")))
print(f"Maximum number of labels attached to a single sample: {train_df['nb_labels'].max()}")
print(f"Minimum number of labels attached to a single sample: {train_df['nb_labels'].min()}")
print("All counts:")
print(train_df["nb_labels"].value_counts())


# Nothing is complete in data analysis without visualizations. Plus, it is the most fun part of an EDA. 

# In[ ]:


plt.figure(figsize=(12,8))
train_df['nb_labels'].value_counts().plot('bar')
plt.show()


# In[ ]:


single_labels_count = train_df[train_df['nb_labels']==1]['nb_labels'].count()
multi_labels_count = train_df[train_df['nb_labels']>1]['nb_labels'].count()

# Plot the value counts for each count
plt.figure(figsize=(12,8))
sns.barplot(x=['Single label', 'Multi-label'], y=[single_labels_count, multi_labels_count])
plt.title("Single vs Multi label distribution", fontsize=16)
plt.xlabel("Label type", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()


# There are a total of 28 labels in our training set. Before ding anything further, create a mapping for the labels. Some people go with the fancy word **"labelmap"**

# In[ ]:


labels_dict={
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}


# One of the things that is always important to check before assuming anything about the data is the distribution of the labels in the training set. Come on, why the hell on this earth we created that label map! For the visualization, of course.

# In[ ]:


# Split the labels
labels = train_df["Target"].apply(lambda x: x.split(" "))

# Create a counter. This initializes the count for each class with a value of zero
labels_count = defaultdict(int)

# Update the counter 
for label in labels:
    if len(labels) > 1:
        for l in label:
            labels_count[labels_dict[int(l)]]+=1
    else:
        labels_count[labels_dict[int(label)]]+=1

# Plot         
plt.figure(figsize=(20,15))
sns.barplot(x=list(labels_count.values()), y=list(labels_count.keys()), color=color[3], orient='h')
plt.show()


# Woah! Huge imbalance. We need to be very careful about this when we are going to a build a model. 

# Each Id in our training set consists of four corresposnding images: Red, Yellow, Blue and Green. (Pardon me, I broke your RGB order here, lol). Now, the ideal thing is to do these steps:
# 
# * Check some images that have only one label.
# * Check some images that consists of more than one label
# * Investigate  further. (Hold on, don't jump on models directly. Stop treating DL as a bazooka without looking at the data first)

# In[ ]:


# Let's see some samples that have single label
sample_indices = train_df.index[train_df['nb_labels']==1][:3]
single_label_samples = train_df['Id'][sample_indices].tolist()
single_label_samples_labels = train_df['Target'][sample_indices].tolist()
single_label_samples_labels = [labels_dict[int(x)] for x in single_label_samples_labels]

f,ax = plt.subplots(3,4, figsize=(40,40), sharex=True, sharey=True)
for i in range(3):
    img_path = str(train_dir / single_label_samples[i])
    red_img = imread(img_path + "_red.png")
    yellow_img = imread(img_path + "_yellow.png")
    blue_img = imread(img_path + "_blue.png")
    green_img = imread(img_path + "_green.png")
    
    ax[i,0].imshow(red_img)
    ax[i,1].imshow(yellow_img)
    ax[i,2].imshow(blue_img)
    ax[i,3].imshow(green_img)
    
    sup_title = single_label_samples_labels[i] + "-" +single_label_samples[i]
    ax[i,0].set_title(sup_title + "_red")
    ax[i,1].set_title(sup_title + "_yellow")
    ax[i,2].set_title(sup_title + "_blue")
    ax[i,3].set_title(sup_title + "_green")
    
plt.show()


# In[ ]:


# Convert labels into a format that can later be used by multilabel binarizer
def get_labels(x):
    labels = x.split(" ")
    labels = [int(x) for x in labels]
    return labels

train_df['labels']= train_df['Target'].apply(get_labels)
train_df.head(10)


# In[ ]:


# Some pseudo-constants
desired_height, desired_width, nb_channels = 224,224,3
nb_classes = len(labels_dict)
images_path = train_dir

# Multilabel binarizer
mlb = MultiLabelBinarizer(classes=np.array(list(labels_dict.keys())))


# In[ ]:


# Custom generator
def data_generator(data, batch_size=16):
    batch_data = np.zeros((batch_size, desired_height, desired_width, nb_channels), dtype=np.float32)
    
    n = len(data)
    steps = n//batch_size
    
    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Initialize a counter
    i =0
    while True: 
        batch_labels = []
        # Get the next batch 
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        
        for j, idx in enumerate(next_batch):
            #print(str(images_path/data.iloc[idx]['Id']))
            img = cv2.imread(str(images_path/data.iloc[idx]['Id']) + "_green.png")
            label = data.iloc[idx]['labels']
            #print(label)
            
            # Resize
            img = cv2.resize(img, (224,224)).astype(np.float32)
            batch_data[j] = img
            batch_labels.append(label)
            
            if j==batch_size-1:
                break     
    
        i+=1
        batch_data = resnet50.preprocess_input(batch_data)
        batch_labels = mlb.fit_transform(batch_labels)
        batch_labels = np.array(batch_labels).astype(np.float32)
        yield batch_data, batch_labels
        del batch_labels
            
        if i>=steps:
            i=0


# In[ ]:


training_data, validation_data = train_test_split(train_df, random_state=seed, test_size=0.2, 
                                                  stratify=train_df['nb_labels'])

print(f"Number of training samples: {len(training_data)}")
print(f"Number of validation samples: {len(validation_data)}")


# In[ ]:


# Reset the index of the two dataframes we got
training_data = training_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)


# In[ ]:


# Generator instances for training anf validation data
train_data_gen = data_generator(training_data)
valid_data_gen = data_generator(validation_data)


# In[ ]:


# Get a pretrained model you want to use 
def get_base_model():
    model = resnet50.ResNet50(input_shape=(224,224,3), include_top=False, weights="imagenet")
    return model


# In[ ]:


base_model = get_base_model()
base_model_output = base_model.output

# Add layers on the top of base model
x = Flatten(name='flat')(base_model_output)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.5, name='dp1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.25, name='dp2')(x)
x = Dense(nb_classes, activation='sigmoid', name='out')(x)

model = Model(inputs=base_model.inputs, outputs=x)
model.summary()


# In[ ]:


opt = RMSprop(lr=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.00001, verbose=1, mode='min')

batch_size = 16
train_steps = len(training_data)//batch_size
valid_steps = len(validation_data)//batch_size


# In[ ]:


results = model.fit_generator(train_data_gen, 
                              steps_per_epoch=train_steps,
                              validation_data=valid_data_gen, 
                              validation_steps=valid_steps,epochs=1, 
                              callbacks=[earlystopper, checkpointer, reduce_lr])


# A few things to which you can contribute:
# * The data generator is slow. Can you make it fast?
# * Accuracy isn't the true metric here. Even in real word, accuracy is of no use in such cases.
# * Validation strategy has to be very clever here
# 
# **Wait for more...I WILL BE BACK!**

# In[ ]:




