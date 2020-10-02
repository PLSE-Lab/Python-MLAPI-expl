#!/usr/bin/env python
# coding: utf-8

# # Peking University/Baidu - Autonomous Driving
# ## Can you predict vehicle angle in different settings?
# ![](https://i.imgur.com/nV7p7ib.png)
# 

# # Data Description
# ## Much of the Text was taken from the official page [here](https://www.kaggle.com/c/pku-autonomous-driving/data)
# 
# This dataset contains photos of streets, taken from the roof of a car. We're attempting to predict the position and orientation of all un-masked cars in the test images. You should also provide a confidence score indicating how sure you are of your prediction.
# 
# Pose Information (train.csv)
# Note that rotation values are angles expressed in radians, relative to the camera.
# 
# - The primary data is images of cars and related `pose` information. The pose information is formatted as strings, as follows:
# `model type, yaw, pitch, roll, x, y, z`
# 
# - A concrete example with two cars in the photo:
# `5 0.5 0.5 0.5 0.0 0.0 0.0 32 0.25 0.25 0.25 0.5 0.4 0.7`
# 
# - Submissions (per sample_submission.csv) are very similar, with the addition of a confidence score, and the removal of the model type. You are not required to predict the model type of the vehicle in question.
# 
# `ID, PredictionString`
# `ID_1d7bc9b31,0.5 0.5 0.5 0.0 0.0 0.0 1.0` indicating that this prediction has a confidence score of 1.0.
# 
# Other Data:
# - **Image Masks (test_masks.zip / train_masks.zip)**
# Some cars in the images are not of interest (too far away, etc.). Binary masks are provided to allow competitors to remove them from consideration.
# 
# - **Car Models**
# 3D models of all cars of interest are available for download as pickle files - they can be compared against cars in images, used as references for rotation, etc.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pickle
import cv2
import json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)


# File descriptions
# - **train.csv** - pose information for all of the images in the training set.
# - **train_images.zip** - images in the training set.
# - **train_masks.zip** - ignore masks for the training set. (Not all images have a mask.)
# - **test_images.zip** - images in the test set.
# - **test_masks.zip** - ignore masks for the test set. (Not all images have a mask.)
# - **sample_submission.csv** - a sample submission file in the correct format
# - **ImageId** - a unique identifier for each image (and related mask, if one exists).
# - **PredictionString** - a collection of poses and confidence scores.
# - **car_models.zip** - 3D models of the unmasked cars in the training / test images. They can be used for pose estimation, etc.
# - **camera.zip** - camera intrinsic parameters.

# In[ ]:


# Look at the data folder
get_ipython().system('ls -GFlash --color ../input/pku-autonomous-driving/')


# # Train csv file
# The train file contains `Pose Information`

# In[ ]:


train = pd.read_csv('../input/pku-autonomous-driving/train.csv')
train.head()


# In[ ]:


print('Example Prediction String....')
print(train['PredictionString'].values[0])


# ## Expanding out the prediction string for the first vehicle
# We know the order of each value in the prediction string. We can expand it out for the first vehicle and see some statistics for this first vehicle position.

# In[ ]:


train_expanded = pd.concat([train, train['PredictionString'].str.split(' ', expand=True)], axis=1)
train_expanded = train_expanded.rename(columns={0 : '1_model_type', 1 : '1_yaw', 2 : '1_pitch',
                                                3 : '1_roll', 4 : '1_x', 5 : '1_y', 6 : '1_z'})
train_expanded.drop('PredictionString', axis=1).head()


# # Training Set, First Car Stats
# 
# - Model type (You are not required to predict the model type of the vehicle in question.)

# In[ ]:


train_expanded.groupby('1_model_type')['ImageId']     .count()     .sort_values()     .plot(kind='barh',
          figsize=(15, 8),
          title='First Car, Count by Model Type',
          color=my_pal[0])
plt.show()


# In[ ]:


train_expanded['1_yaw'] = pd.to_numeric(train_expanded['1_yaw'])
train_expanded['1_yaw']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car YAW',
          color=my_pal[1])
plt.show()


# In[ ]:


train_expanded['1_pitch'] = pd.to_numeric(train_expanded['1_pitch'])
train_expanded['1_pitch']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car pitch',
          color=my_pal[2])
plt.show()


# In[ ]:


train_expanded['1_roll'] = pd.to_numeric(train_expanded['1_roll'])
train_expanded['1_roll']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car roll',
          color=my_pal[3])
plt.show()


# ## X, Y, and Z features

# In[ ]:


train_expanded['1_x'] = pd.to_numeric(train_expanded['1_x'])
train_expanded['1_y'] = pd.to_numeric(train_expanded['1_y'])
train_expanded['1_z'] = pd.to_numeric(train_expanded['1_z'])
train_expanded['1_x']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of x',
          color=my_pal[0])
plt.show()
train_expanded['1_y']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of y',
          color=my_pal[1])
plt.show()
train_expanded['1_z']     .dropna()     .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of z',
          color=my_pal[2])
plt.show()


# # Sample submission

# In[ ]:


ss = pd.read_csv('../input/pku-autonomous-driving/sample_submission.csv')
ss.head()


# # Reading Images
# - Train and test images and masks are each in their own folder.
# - Lets look at the image with the most cars from the training set.
# - All are jpg files

# In[ ]:


# Lets look at the first few images on disk
get_ipython().system('ls -GFlash ../input/pku-autonomous-driving/train_images | head')


# ## Image Example

# In[ ]:


plt.rcParams["axes.grid"] = False

train_ids = train['ImageId'].values
img_name = train.loc[2742]['ImageId']
fig, ax = plt.subplots(figsize=(15, 15))
img = load_img('../input/pku-autonomous-driving/train_images/' + img_name + '.jpg')
plt.imshow(img)
plt.show()


# ## Mask Example

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
mask = load_img('../input/pku-autonomous-driving/train_masks/' + img_name + '.jpg')
plt.imshow(mask)
plt.show()


# ## Plotting Mask over the Images

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(img)
plt.imshow(mask, cmap=plt.cm.viridis, interpolation='none', alpha=0.5)
plt.show()


# ## Masks Next to Images

# In[ ]:


ids = train['ImageId'].values
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
for i in range(4):
    img = load_img('../input/pku-autonomous-driving/train_images/' + ids[i] + '.jpg')
    img_mask = load_img('../input/pku-autonomous-driving/train_masks/' + ids[i] + '.jpg')
    #plt.subplot(1,2*(1+len(ids)),q*2-1)
    ax=axes[i][0].imshow(img)
    #plt.subplot(1,2*(1+len(ids)),q*2)
    ax=axes[i][1].imshow(img_mask)
    ax=axes[i][2].imshow(img)
    ax=axes[i][2].imshow(img_mask, cmap=plt.cm.viridis, interpolation='none', alpha=0.4)
plt.show()


# ## Camera intrinsic parameters.

# In[ ]:


get_ipython().system('cat ../input/pku-autonomous-driving/camera/camera_intrinsic.txt')


# ## 3D Car Models

# In[ ]:


get_ipython().system('ls -GFlash ../input/pku-autonomous-driving/car_models/ | head')


# Per the data description
# *3D models of all cars of interest are available for download as pickle files - they can be compared against cars in images, used as references for rotation, etc.*
# 
# The pickles were created in Python 2. For Python 3 users, the following code will load a given model:
# ```
# with open(model, "rb") as file:
#     pickle.load(file, encoding="latin1")
# ```
# 
# **This doesn't appear to work on kaggle kernels, returns a `ModuleNotFoundError: No module named 'objloader'`**

# In[ ]:


# model = '../input/pku-autonomous-driving/car_models/aodi-Q7-SUV.pkl'
# with open(model, "rb") as file:
#     pickle.load(file, encoding="latin1")


# ## 3D Car Model JSON Files
# We can however load the json files located in the `../input/pku-autonomous-driving/car_models_json/` directory

# In[ ]:


get_ipython().system('ls -GFlash ../input/pku-autonomous-driving/car_models_json/ | head')


# In[ ]:


with open('../input/pku-autonomous-driving/car_models_json/mazida-6-2015.json') as json_file:
    car_model_data = json.load(json_file)


# The File contains the car type, vertices and faces.

# In[ ]:


for keys in enumerate(car_model_data):
    print(keys)


# In[ ]:


def plot_3d_car(model_json_file):
    with open(f'../input/pku-autonomous-driving/car_models_json/{model_json_file}') as json_file:
        car_model_data = json.load(json_file)

    vertices = np.array(car_model_data['vertices'])
    faces = np.array(car_model_data['faces']) - 1
    car_type = car_model_data['car_type']
    x, y, z = vertices[:,0], vertices[:,2], -vertices[:,1]
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(30, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(60, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(-20, 180)
    plt.show()
    return


# ## MG GT
# ![](https://i.ytimg.com/vi/tOG-EYjjyS0/maxresdefault.jpg)

# In[ ]:


plot_3d_car('MG-GT-2015.json')


# ## Aodi Q7
# ![](https://cars.usnews.com/static/images/Auto/izmo/i108385760/2019_audi_q7_angularfront.jpg)

# In[ ]:


plot_3d_car('aodi-Q7-SUV.json')


# # Mazia 6
# ![](https://static.carsdn.co/cldstatic/wp-content/uploads/img-1360163241-1554498653476.jpg)

# In[ ]:


plot_3d_car('mazida-6-2015.json')

