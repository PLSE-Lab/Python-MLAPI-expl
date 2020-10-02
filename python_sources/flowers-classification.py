#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The model is designed to classify flowers by species.
# 
# Our EDA is structured in the following way:
# 
# **[First](#step1)**: Data analysis.
# 
# **[Second](#step2)**:  Model training.
# 
# **[Third](#step3)**: CNN Model training.
# 
# **[Fourth](#step4)**: CNN using transfer learning.
# 

# ---
# <a id='step1'></a>
# ## Step 1: Data Preprocessing
# 
# ### Import Libraries
# Here we import a set of useful libraries

# In[ ]:


from glob import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


from keras.utils import np_utils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers/flowers"))

# Any results you write to the current directory are saved as output.


# ### Analyze the data
# Data is stored in a directory named "flower", in seperate sub-directories based on the species.

# The flowers are represented in dataset as follows: 
# 
# ```
#                                                        flowers
# 
# Daisy                     Dandelion                     Rose                     Sunflower                     Tulip
# 
# ```
# We can split dataset in order to produce training, validation and testing sets that would easily utilize `load_files` from `sklearn`
# 
# ```
#                                                                                                                           data
#                                                         training                                                        validation                                                        test
#                                   Daisy  |  Dandelion  |  Rose  |  Sunflower  |  Tulip      &&     Daisy  |  Dandelion  |  Rose  |  Sunflower  |  Tulip     &&    Daisy  |  Dandelion  |  Rose  |  Sunflower  |  Tulip
# 
# ```

# In[ ]:


# Make a parent directory `data` and three sub directories `train`, `valid` and 'test'
get_ipython().run_line_magic('rm', '-rf data # Remove if already present')

get_ipython().run_line_magic('mkdir', '-p data/train/daisy')
get_ipython().run_line_magic('mkdir', '-p data/train/tulip')
get_ipython().run_line_magic('mkdir', '-p data/train/sunflower')
get_ipython().run_line_magic('mkdir', '-p data/train/rose')
get_ipython().run_line_magic('mkdir', '-p data/train/dandelion')

get_ipython().run_line_magic('mkdir', '-p data/valid/daisy')
get_ipython().run_line_magic('mkdir', '-p data/valid/tulip')
get_ipython().run_line_magic('mkdir', '-p data/valid/sunflower')
get_ipython().run_line_magic('mkdir', '-p data/valid/rose')
get_ipython().run_line_magic('mkdir', '-p data/valid/dandelion')

get_ipython().run_line_magic('mkdir', '-p data/test/daisy')
get_ipython().run_line_magic('mkdir', '-p data/test/tulip')
get_ipython().run_line_magic('mkdir', '-p data/test/sunflower')
get_ipython().run_line_magic('mkdir', '-p data/test/rose')
get_ipython().run_line_magic('mkdir', '-p data/test/dandelion')


get_ipython().run_line_magic('ls', 'data/train')
get_ipython().run_line_magic('ls', 'data/valid')
get_ipython().run_line_magic('ls', 'data/test')


# Find all the categories of the flowers

# In[ ]:


base_dir = "../input/flowers/flowers"
categories = os.listdir(base_dir)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from shutil import copyfile

plt.rcParams["figure.figsize"] = (20,3)


# In[ ]:


def train_valid_test(files):
    """This function splits the files in training, validation and testing sets with 60%, 20%
    and 20% of data in each respectively"""
    train_fles = files[:int(len(files)*0.6)]
    valid_files = files[int(len(files)*0.6):int(len(files)*0.8)]
    test_files = files[int(len(files)*0.8):]
    return train_fles, valid_files, test_files


# In[ ]:


def copy_files(files, src, dest):
    """This function copy files from src to dest"""
    for file in files:
        copyfile("{}/{}".format(src, file), "{}/{}".format(dest, file))


# In[ ]:


def plot_images(category, images):
    """This method plots five images from a category"""
    for i in range(len(images)):
        plt.subplot(1,5,i+1)
        plt.title(category)
        image = mpimg.imread("{}/{}/{}".format(base_dir, category, images[i]))
        plt.imshow(image)
    plt.show()


# In[ ]:


total_images = []
for category in categories:
    images = os.listdir("{}/{}".format(base_dir, category))
    random.shuffle(images)
    filtered_images = [image for image in images if image not in ['flickr.py', 'flickr.pyc', 'run_me.py']]
    
    total_images.append(len(filtered_images))
    
    
    train_images, valid_images, test_images = train_valid_test(filtered_images)
    
    copy_files(train_images, "{}/{}".format(base_dir, category), "./data/train/{}".format(category))
    copy_files(valid_images, "{}/{}".format(base_dir, category), "./data/valid/{}".format(category))
    copy_files(test_images, "{}/{}".format(base_dir, category), "./data/test/{}".format(category))
    plot_images(category, images[:5])
    
        


# ### Statistics of flowers

# In[ ]:


print("Total images: {}".format(np.sum(total_images)))
for i in range(len(categories)):
    print("{}: {}".format(categories[i], total_images[i]))


# In[ ]:


y_pos = np.arange(len(categories))
plt.bar(y_pos, total_images, width=0.2,color='b',align='center')
plt.xticks(y_pos, categories)
plt.ylabel("Image count")
plt.title("Image count in different categories")
plt.show()


# ### Observations
# - There are 4323 total images with approximately similar distribution in each category.
# - The dataset does not seem  to be imbalanced.
# - Accuracy can be used as a metric for model evaulation.

# In[ ]:


# define function to load train, valid and test datasets
def load_dataset(path):
    data = load_files(path)
    flower_files = np.array(data['filenames'])
    print(data['target_names'])
    flower_targets = np_utils.to_categorical(np.array(data['target']), 5)
    return flower_files, flower_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

print('There are %d total flower categories.' % len(categories))
print('There are %s total flower images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training flower images.' % len(train_files))
print('There are %d validation flower images.' % len(valid_files))
print('There are %d test flower images.' % len(test_files))


# ### Data Transformation
# 
# Keras' CNNs require a 4D tensor as input with the shape as `(nb_samples, rows, columns, channels)` where
# - `nb_samples`: total number of samples or images
# - `rows`: number of rows of each image
# - `columns`: number of columns of each image
# - `channels`: number of channels of each image
# 

# In[ ]:


from keras.preprocessing import image                  
from tqdm import tqdm


# ### Create a 4D tensor
# The `path_to_tensor` function below takes a color image as input and returns a 4D tensor suitable for supplying to Keras CNN. The function first loads the image and then resizes it 224x224 pixels. The image then, is converted to an array and resized to a 4D tensor. The returned tensor will always have a shape of `(1, 224, 224, 3)` as we are dealing with a single image only in this function.

# In[ ]:


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# The `ptahs_to_tensor` applies `path_to_tensor` to all images and returns a list of tensors.

# In[ ]:


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# ### Pre-process the Data
# Rescale the images by dividing every pixel in every image by 255.

# In[ ]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# <a id="step2"></a>
# ## Step 2: Develop a Benchmark model
# Use a simple CNN to create a benchmark model.

# In[ ]:


simple_model = Sequential()
print(train_tensors.shape)

### Define the architecture of the simple model.
simple_model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
simple_model.add(GlobalAveragePooling2D())
simple_model.add(Dense(5, activation='softmax'))
simple_model.summary()


# ### Making Predictions with the simple model

# In[ ]:


simple_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Create a `saved_models` directory for saving best model
get_ipython().run_line_magic('mkdir', '-p saved_models')


# In[ ]:


from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.simple.hdf5', 
                               verbose=1, save_best_only=True)

simple_model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[ ]:


simple_model.load_weights('saved_models/weights.best.simple.hdf5')


# In[ ]:


# get index of predicted flower category for each image in test set
flower_predictions = [np.argmax(simple_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### Benchmark model's performance
# The accuracy obtained from the benchmark model is 41.57%.

# ---
# <a id="step3"></a>
# ## Step 3: Develop a CNN architecture from scratch

# In[ ]:


model = Sequential()
print(train_tensors.shape)
### Define architecture.
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# #### Load best weight of the model

# In[ ]:


model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# #### Get the accuracy of the model

# In[ ]:


# get index of predicted flower category for each image in test set
flower_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# <a id="step4"></a>
# ## Step 4: Develop a CNN using Transfer Learning

# In[ ]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_shape=(224,224,3))
for layer in inception_resnet.layers[:5]:
    layer.trainable = False

output_model = inception_resnet.output
output_model = Flatten()(output_model)
output_model = Dense(200, activation='relu')(output_model)
output_model = Dropout(0.5)(output_model)
output_model = Dense(200, activation='relu')(output_model)
output_model = Dense(5, activation='softmax')(output_model)

model = Model(inputs=inception_resnet.input, outputs=output_model)
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inception_resnetv2.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# #### Load the best weight of the model

# In[ ]:


### load best weights
model.load_weights('saved_models/weights.best.inception_resnetv2.hdf5')


# #### Get the accuracy on test set

# In[ ]:


# get index of predicted flower category for each image in test set 
flower_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# In[ ]:


for i in range(5):
    predicted = np.argmax(model.predict(np.expand_dims(test_tensors[i], axis=0)))
    actual = np.argmax(test_targets[i])
    print("Predicted: {}, Actual: {}, Name: {}".format(predicted, actual, test_files[i].split("/")[2]))
    image = mpimg.imread(test_files[i])
    plt.imshow(image)
    plt.show()


# In[ ]:




