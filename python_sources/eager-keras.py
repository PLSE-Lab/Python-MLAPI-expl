#!/usr/bin/env python
# coding: utf-8

# Hello everyone. I am gonna do something very crazy today. In my last [kernel](https://www.kaggle.com/aakashnain/eagerfmnist) on Eager mode in tensorflow, people asked me to do a kernel on the [tf.dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Well, I actually wanted to do one but doing a simple kernel on dataset API wasn't something motivating to invest my time on a kernel. So, I thought about it and then I came up with a thought to show that how can you mix `Keras`, `Eager`, `TF` and `tf.data.Dataset` API, all in one single model!! 
# 
# I am gonna show a lot of cool stuff today(it was cool for me atleast :P). It's a bit long kernel as compared to others, so hold tight. I hope this will kernel will help fine grain your understanding on the above four things. 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import cv2
import glob
import shutil
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(111)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I have added VGG16 weights as the external data source to this kernel because we will be doing transfer learning in the end. 

# In[2]:


# Import tensorflow 
import tensorflow as tf
# Enable eager execution
import tensorflow.contrib.eager as tfe
# I will explain this thing in the end
tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
# Import keras but from inside tf 
from tensorflow import keras


# In[3]:


# We can check if we are executing in eager mode or not. This code should print true if we are in eager mode.
tf.executing_eagerly()


# We will make a directory  to store VGG weight for keras. I have already shown how to do this  in one of my kernels. If you want to get more details on this step, have a look at [this](https://www.kaggle.com/aakashnain/flowers-are-mesmerizing)

# In[4]:


# Check for the directory and if it doesn't exist, make one.
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
    
# make the models sub-directory
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


# In[5]:


# Copy the weight to the .keras/models directory
get_ipython().system('cp ../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/')


# In[6]:


# As usual, define some paths first to make life simpler
training_data = Path('../input/10-monkey-species/training/training/') 
validation_data = Path('../input/10-monkey-species/validation/validation/') 
labels_path = Path('../input/10-monkey-species/monkey_labels.txt')


# Information about the labels are given in the `monkey_labels.txt` file. This file contains `label, latin name, common name, corresponding train images count, corresponding validation images count`. Let's read this file and store this information in a pandas dataframe.

# ## Simple analysis

# In[ ]:


# How many categories are there in the dataset?
labels_info = []

# Read the file
lines = labels_path.read_text().strip().splitlines()[1:]
for line in lines:
    line = line.split(',')
    line = [x.strip(' \n\t\r') for x in line]
    line[3], line[4] = int(line[3]), int(line[4])
    line = tuple(line)
    labels_info.append(line)
    
labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images'], index=None)
labels_info.head(10)


# You can see from the above output that the number of images for each category is almost similar which is a good thing this almost a balanced dataset. We can plot this information too.

# In[8]:


labels_info.plot(x='Label', y=['Train Images','Validation Images'], kind='bar', figsize=(20,5))
plt.ylabel('Count of images')
plt.show()


# We will be training our model on random batches of the training dataset.  As the training images are already provided in separate sub-directories, `keras ImageDataGenerator` would have been a perfect choice but as we are going to use the `tf.data.Dataset` API, we will read the file names and corresponding labels and we will store them in a pandas dataframe.

# ## Preprocessing 

# In[9]:


# Creating a dataframe for the training dataset
train_df = []
for folder in os.listdir(training_data):
    # Define the path to the images
    imgs_path = training_data / folder
    # Get the list of all the images stored in that directory
    imgs = sorted(imgs_path.glob('*.jpg'))
    print("Total number of training images found in the directory {}: {}".format(folder, len(imgs)))
    # Append the info to out list for training data 
    for img_name in imgs:
        train_df.append((str(img_name), folder))

train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)        
print("\n",train_df.head(10), "\n")
print("=================================================================\n")


# Creating dataframe for validation data in a similar fashion
valid_df = []
for folder in os.listdir(validation_data):
    imgs_path = validation_data / folder
    imgs = sorted(imgs_path.glob('*.jpg'))
    print("Total number of validation images found in the directory {}: {}".format(folder, len(imgs)))
    for img_name in imgs:
        valid_df.append((str(img_name), folder))

valid_df = pd.DataFrame(valid_df, columns=['image', 'label'], index=None)        
print("\n", valid_df.head(10))


# In[10]:


# Shuffle the train and validation dataframes
train_df = train_df.sample(frac=1.).reset_index(drop=True)
valid_df = valid_df.sample(frac=1.).reset_index(drop=True)

print("Total number of training samples: ", len(train_df))
print("Total number of validation samples: ", len(valid_df))


# In[11]:


# Create a dictionary to map the labels to integers
labels_dict= {'n0':1, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'n5':5, 'n6':6, 'n7':7, 'n8':8, 'n9':9}


# In[13]:


# Let's look at some sample images first
sample_images = []
sample_train_images_df = train_df[:12]
f,ax = plt.subplots(4,3, figsize=(30,30))
for i in range(12):
    img = cv2.imread(sample_train_images_df.loc[i, 'image'])
    img = cv2.resize(img, (224,224))
    inv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = labels_info.loc[labels_info['Label'] == sample_train_images_df.loc[i, 'label']]['Common Name'].values[0]
    sample_images.append(img)
    ax[i//3, i%3].imshow(inv_img)
    ax[i//3, i%3].set_title(lab, fontsize=14)
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_aspect('auto')
plt.show() 


# Now, it's time to define our batch data generator using tf.Dataset API. There are many ways in which you can intialize an iterator instance of the `tf.data.Dataset` API. For example:
# * You can read tensor slices from numpy arrays, given that your data can fit in the memory
# * You can read tensor slices from TFRecords
# * You can read tensor slices from filenames, etc.
# 
# I highly recommend going over the documentation over [reading inputs](https://www.tensorflow.org/programmers_guide/datasets#reading_input_data). This will help you understand how you should initialize your input pipeline for your model.
# 
# A question that you must ask: **Why use this API? Why not simply define a python generator ?**
# This is a very good question. When you train your model, you read data from disk in batches and then the data is transferred to GPU for training. Plus GPU operates much faster and do heavily parallized operations as compared to CPU.  Hence most of the time, GPU sits idle as processing data on CPU and transferring it to GPU after ward is a bottleneck. The `tf.data.Dataset` API tries to overcome this limitation by processing data in a much more efficient way. It follows the ETL methodology. I can give more details about it but it would be better if you just watch [this](https://youtu.be/uIcqeP7MFH0?list=PLQY2H8rRoyvxjVx3zfw4vA4cvlKogyLNN) video from tf summit. **I highly recommend it.**
# 
# P.S: There are a lot of features that are still not documented properly on the TensorFlow page. I spent a lot of time figuring out the best way to use it but phew!!!! Similarly, the documentation for eager mode is a bit sparse at this moment but nothing is stopping you to experiment with all these things.

# In[61]:


# A data generator using tf dataset
def data_gen(X=None, y=None, batch_size=32, nb_epochs=1):
    # A simple function to decode an image
    # You can use cv2 also but for that the syntax is a little bit ugly this time, so we will use this as of now
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        # We will resize each image to 224x224x3
        image_resized = tf.cast(tf.image.resize_images(image_decoded, [224, 224]), dtype=tf.float32)
        #Normalize the pixel values
        image_resized = tf.truediv(image_resized, 255.)
        # One hot encoding of labels
        label = tf.one_hot(label, depth=10, dtype=tf.float32)
        return image_resized, label
    
    # We will read tensor slices from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    # This num_parallel_calls is for executing this functions on multiple cores
    dataset = dataset.map(_parse_function,num_parallel_calls=32)
    dataset = dataset.batch(batch_size)#.repeat(nb_epochs)
    return dataset


# In order to use the dataset, we will get a list of all image files and a corresponding list of image labels. 

# In[29]:


# Get a list of all the images and the corresponding labels in the training data
train_images = tf.constant(train_df['image'].values)
train_labels = tf.constant([labels_dict[l] for l in train_df['label'].values])

# Do the same for the validation data
valid_images = tf.constant(valid_df['image'].values)
valid_labels = tf.constant([labels_dict[l] for l in valid_df['label'].values])


# We will start defining our model now. We will use the VGG16 pre trained network as our base network(up to conv block) and then add some dense layers on the top of that. As it's transfer learning, we will freeze the weights of the base network and will train our newly added layers only.

# ## Model

# In[57]:


# Model definition
def build_model():
    base_model = keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model_output = base_model.output
    x = keras.layers.Flatten(name='flatten')(base_model_output)
    x = keras.layers.Dense(1024, activation='relu', name='fc1')(x)
    x = keras.layers.Dropout(0.5, name='dropout1')(x)
    x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = keras.layers.Dropout(0.5, name='dropout2')(x)
    out = keras.layers.Dense(10, name='output', activation='softmax')(x)
    
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.models.Model(inputs=base_model.input, outputs=out)
    return model


# In[63]:


model = build_model()
model.summary()


# In[64]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001))


# We are all set to train our model. We require to do the following things:
# * Define batch size
# * Define number of epochs
# * Get an instance of our data generator
# * Train model on the incoming batch
# * Calculate  batch loss and batch accurcay

# ## Training and Evaluation

# In[70]:


# Number of epochs for which you want to train your model
nb_epochs = 2

# Define a batch size you want to use for training
batch_size=32

# Get an instance of your iterator
train_gen = data_gen(X=train_images, y=train_labels, batch_size=batch_size)
valid_gen = data_gen(X=valid_images, y=valid_labels, batch_size=batch_size)


# Number of training and validation steps in an epoch
nb_train_steps = train_images.shape.num_elements() // batch_size
nb_valid_steps = valid_images.shape.num_elements() // batch_size
print("Number of training steps: ", nb_train_steps)
print("Number of validation steps: ", nb_valid_steps)

with tf.device('/GPU:0'):
    for epoch in range(nb_epochs):
        train_loss, train_acc= [], []
        for (images, labels) in tfe.Iterator(train_gen):
            loss, accuracy = model.train_on_batch(images.numpy(), labels.numpy())
            train_loss.append(loss.numpy())
            train_acc.append(accuracy.numpy())    
        print("======================= Epoch {} ===================================== ".format(epoch))
        print("train_loss: {:.2f}    train_acc: {:.2f}".format(np.mean(train_loss), np.mean(train_acc)*100))


# Simple!!  isn't it? I have tried my best to show that how can you mix tf, keras, eager, and tf.data and make the best use out of it. There are many more things that you can do. 
# For example, you can create your own network using the **tf.layers** API and use all the functionality of keras inside that. In order to do that make sure that your model class inherits the `keras.model`

# ## Data Augmentation
# Data augmentation is a crucial part of training deep learning models. Writing an efficient augmentation pipeline is itself a big task. Keras provide `ImageDataGenerator` class that can do augmentation on the fly and it's very good but there are certain number of cases when you want to use  some other augmentation techniques.  Although you can write your own augmentation pipeline but I recommend using `imgaug` phython package. It's very flexible and easy to use. Lemme show you a dummy example for the same.

# In[71]:


# We don't want all the augmentations to be applied on the iamge. The code line below tells that anything isnide iaa.Sometime()
# will be applied only to random 50% of the images  
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Different types if sugmentation you want to apply on your batch of images
aug_to_apply = [iaa.Fliplr(0.2), sometimes(iaa.Affine(rotate=(10,30))), 
                sometimes(iaa.Multiply((0.5, 1.5)))]
# Instantiate the augmenter
aug = iaa.Sequential(aug_to_apply, random_order=True, random_state=111)


# In[74]:


# I will do the augmentation on the images that I showed above

# Get the numpy array [batch, img.shape]
sample_images_arr = np.array(sample_images).reshape(len(sample_images), 224, 224, 3)

# Augment the images using augmenter instance we just created in the above cell
images_aug = aug.augment_images(sample_images_arr)
images_aug = images_aug[...,::-1]


# In[75]:


f,ax = plt.subplots(4,3, figsize=(30,30))
for i in range(12):
    img = images_aug[i]
    lab = labels_info.loc[labels_info['Label'] == sample_train_images_df.loc[i, 'label']]['Common Name'].values[0]
    ax[i//3, i%3].imshow(img)
    ax[i//3, i%3].set_title(lab, fontsize=14)
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_aspect('auto')
plt.show()    


# ## Final Thoughts
# * Eager and keras are a great way to go but the documentation at the moment is very unclear. I heard that the it's in process, so fingers crossed.
# * There is a particular functionality `prefecth_to_device()` in the `dataset` API which actually gives the real boost in performance but at the moment there is a bug in that and that's why we use `tfe.Iterator()`
# * IMHO, TF 1.8 was released too early. I have been doing many things since last two weeks and so many things broke. A proper testing of the new functionalities would have been better 
# *  One of the most annoying thing with eager as of now is the device placement of TF ops. For eager it is required that the ops are on the same device(either CPU/GPU) to work. That's why we put this line `tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)` in the beginning. This is totally insane!! Why? Because it will downside all the boost in performance we get using `tf.data.Dataset`. Device placement is a very expensive operation and is a performance bottleneck. TF guys need to fix these things  ASAP because if they don't, then there will be no eager. No one wants to waste their time on the bugs when they already have pure tf, keras or pytorch to experiment on.

# That's it folks!! I hope you enjoyed every bit of this kernel. **Please upvote if you liked the kernel.** Suggestion/Feedback is welcome.   

# In[ ]:




