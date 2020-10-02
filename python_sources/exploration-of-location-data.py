#!/usr/bin/env python
# coding: utf-8

# # Exploration of Scotland image location data 
# 
# 
# Within this notebook we start by performing some basic extraction of EXIF formatted data from all the images within the dataset. From this, various image meta-data features are explored, which help to plot the location of all images taken on a map of Scotland. Subsequently, a dataset of images and their associated town or city is created, which then allows the formation of a model to make predictions of the town in which an image was taken.
# 
# Two different Deep Convolutional Neural Network models are formed to perform these predictions. Each model predict the town in which an image was taken based entirely on the visual features of a given image. 
# 
# The first model is a Simple convolutional neural network architecture, whilst the second is more complex and is formed through transfer learning and feature extraction from a pre-trained Xception imagenet model.
# 
# The intention of this work was to evaluate the feasibility of using a deep neural convolutional neural network to classify the town / city of an image based entirely on its visual features. The task was shown to be extremely difficult, which is suspected to be due to a limitation in the number of images to perform this task effectively. However, given enough data, this task could be much more feasible with higher performing models that can generalise to more locations.

# In[ ]:


get_ipython().system('pip install reverse-geocode')


# In[ ]:


import folium
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import reverse_geocode
import seaborn as sns
import shutil

from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from pathlib import Path
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image

from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import exposure

from sklearn.model_selection import train_test_split


# In[ ]:


# if using gpu - confirm
import tensorflow as tf
tf.test.gpu_device_name()


# Set random seed for the work used throughout this notebook:

# In[ ]:


np.random.seed(1)
tf.random.set_seed(1)


# ---
# ## 1. Dataset exploration and pre-processing

# ### Extract EXIF metadata from all images and form a dataset

# In[ ]:


def extract_exif(filename):
    """ Extract img EXIF data """
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def extract_exif_labelled(exif_data):
    """ Extract EXIF data with formatted labels """
    labelled_data = {}
    for (key, val) in exif_data.items():
        labelled_data[TAGS.get(key)] = val
    return labelled_data


def extract_geotags(exif_data):
    """ Obtain better formatted geotag data """
    if not exif_data:
        raise ValueError("EXIF metadata not found.")
    geotags = {}
    for (idx, geotag) in TAGS.items():
        if geotag == 'GPSInfo':
            if idx not in exif_data:
                raise ValueError("No EXIF geotagging found")
            for (key, val) in GPSTAGS.items():
                if key in exif_data[idx]:
                    geotags[val] = exif_data[idx][key]
    return geotags


def lat_long_alt_from_geotag(geotags):
    """ Obtain decimal lat, long and altitude from geotags """
    lat = dms_to_decimal(geotags['GPSLatitudeRef'], geotags['GPSLatitude'])
    long = dms_to_decimal(geotags['GPSLongitudeRef'], geotags['GPSLongitude'])
    
    # obtain altitude data and process, if it exists
    altitude = None
    try:
        alt = geotags['GPSAltitude']
        altitude = alt[0] / alt[1]
        
        # multiple by -1 if below sea level
        if geotags['GPSAltitudeRef'] == 1: 
            altitude *= -1
    except KeyError:
        altitude = 0
  
    return lat, long, altitude


def dms_to_decimal(lat_long_ref, deg_min_sec):
    """ Convert degrees, minutes, seconds tuples into decimal
        lat and lon values. Given to 5 decimal places - more 
        than sufficient for commercial GPS """
    
    degrees = deg_min_sec[0][0] / deg_min_sec[0][1]
    minutes = deg_min_sec[1][0] / deg_min_sec[1][1] / 60.0
    seconds = deg_min_sec[2][0] / deg_min_sec[2][1] / 3600.0
    
    if lat_long_ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds
        
    return round(degrees + minutes + seconds, 5)


# #### Choose an image as an example, and display metadata

# In[ ]:


example_filename = '/kaggle/input/geolocated-imagery-dataset-scotland/300-399/322.jpg'
exif = extract_exif(example_filename)
labeled = extract_exif_labelled(exif)

for key, val in labeled.items():
    print(f"{key} : {val}")


# In[ ]:


geo_data = extract_geotags(exif)

for key, val in geo_data.items():
    print(f"{key} : {val}")


# In[ ]:


coords = lat_long_alt_from_geotag(geo_data)
print(coords)


# ### Extract EXIF metadata including GPS locations and other details from all images

# In[ ]:


img_name, img_path = [], []
latitudes, longitudes, altitudes = [], [], []
img_width, img_height = [], []
makes, models = [], []
time_dates = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    
    for filename in filenames:
        
        if filename.endswith('.jpg'):
        
            file_path = os.path.join(dirname, filename)
        
            exif_data = extract_exif(file_path)
            exif_labels = extract_exif_labelled(exif_data)
            geo_data = extract_geotags(exif_data)
            lat, long, alt = lat_long_alt_from_geotag(geo_data)
        
            img_name.append(filename)
            img_path.append(file_path)
            latitudes.append(lat)
            longitudes.append(long)
            altitudes.append(alt)
            img_width.append(exif_labels.get('ImageWidth', 0))
            img_height.append(exif_labels.get('ImageLength', 0))
            makes.append(exif_labels.get('Make', 'Unknown'))
            models.append(exif_labels.get('Model', 'Unknown'))
            time_dates.append(exif_labels.get('DateTime', 0))


# #### Determine city for each coordinate

# In[ ]:


# reverse geocode our coordinates for the city
coord_pairs = [(lat,long) for lat, long in zip(latitudes, longitudes)]
cities = [x['city'] for x in reverse_geocode.search(coord_pairs)]


# #### Form dataframe for all our imagery metadata

# In[ ]:


metadata_df = pd.DataFrame({ 'filename' : img_name, 'filepath' : img_path, 
                             'img_width' : img_width, 'img_height' : img_height, 
                             'make' :makes, 'model' : models,
                             'latitude' : latitudes, 'longitude' : longitudes, 
                             'altitude' : altitudes, 'time_date' : time_dates,
                             'city' : cities})

metadata_df.head()


# In[ ]:


metadata_df['img_height'].value_counts()


# #### Quick visualisation of dataset features for interest
# 
# Altitude of the images taken in the dataset:

# In[ ]:


plt.figure(figsize=(10,5))
metadata_df['altitude'].plot()
plt.ylabel("Altitude (m)")
plt.xlabel("Image number")
plt.show()


# Sort dataset by date and visualise change in altitude as each image was taken:

# In[ ]:


date_index_df = metadata_df.copy()
date_index_df['Date'] = pd.to_datetime(date_index_df['time_date'], format='%Y:%m:%d %H:%M:%S')
date_index_df.sort_values(by=['Date'], inplace=True, ascending=True)
date_index_df.reset_index(inplace=True, drop=True)

plt.figure(figsize=(10,5))
date_index_df['altitude'].plot()
plt.ylabel("Altitude (m)")
plt.xlabel("Image number")
plt.show()


# Visualise the number of images taken in each town / city:

# In[ ]:


plt.figure(figsize=(12,6))
values = metadata_df['city'].value_counts()
sns.barplot(x = values.index.values, y = values.values)
plt.xticks(rotation=90)
plt.show()


# Lets focus on just the top ten towns / cities that have a reasonable number of samples (20+):

# In[ ]:


# obtain towns / cities with top ten image counts
values = metadata_df['city'].value_counts()[:10]

plt.figure(figsize=(10,5))
sns.barplot(x = values.index.values, y = values.values)
plt.xticks(rotation=90)
plt.show()

top_towns = list(values.index.values)
top_towns


# Caol is located near Fort William and as such is very picturesque - no wonder the most images throughout the dataset were taken here!
# 
# With the large number of towns/cities with insignificant numbers of images when compared to the top ten (greater than 20 images), it is worth only selecting the top n cities with a similar number of images. This avoids biasing the training of our dataset.

# #### Visualise an example image from each of the top ten towns / cities

# In[ ]:


fig = plt.figure(figsize=(12, 6))

for i, example in enumerate(top_towns):
    
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    class_imgs = metadata_df[metadata_df['city'] == example]
    example_img_path = class_imgs.iloc[0]['filepath']
    
    example_img = imread(example_img_path)
    
    ax.imshow(example_img)
    
    ax.set_xlabel(example)


# In the following section, we'll plot the locations of all images on a map of Scotland to give us an appreciation of the variation of geographical images we have within the dataset.

# ### Annotate image locations on a map of Scotland using Folium

# ###### Plot all points, regardless of location:

# In[ ]:


def plot_data_coords(row):
    folium.Circle(location=[row.latitude, row.longitude],
                  color='crimson',
                  tooltip = "<h5 style='text-align:center;font-weight: bold'>Img Name : "+row.filename+"</h5>"+
                            "<hr style='margin:10px;'>"+
                            "<ul style='color: #444;list-style-type:circle;align-item:left;"+
                            "padding-left:20px;padding-right:20px'>"+
                            "<li>Town : "+str(row.city)+"</li>"+
                            "<li>Lat : "+str(row.latitude)+"</li>"+
                            "<li>Long : "+str(row.longitude)+"</li>"+
                            "<li>Altitude : "+str(row.altitude)+"</li>"+
                            "<li>Time date : "+str(row.time_date)+"</li></ul>",
                  radius=20, weight=6).add_to(m)

    
m = folium.Map(location=[metadata_df['latitude'].mean(), 
                         metadata_df['longitude'].mean()], 
               tiles='OpenStreetMap',
               min_zoom=7, max_zoom=12, zoom_start=7.5)


# iterate through all rows and plot coords
metadata_df.apply(plot_data_coords, axis = 1)

m


# ###### Plot top ten town / city locations, with a different colour for each on the map:

# In[ ]:


colors = ['red', 'blue', 'gray', 'darkred', 'black', 'orange', 'beige', 'green', 
          'purple', 'lightgreen', 'darkblue', 'lightblue', 'darkgreen', 'darkpurple',
          'lightred', 'cadetblue', 'lightgray', 'pink']

# dict comp to form unique color for each town
town_colors = { town : color for town, color in zip(top_towns, colors[:len(top_towns)]) }

# select only data containing our selected towns / cities
top_towns_df = metadata_df[metadata_df['city'].isin(top_towns)]


# In[ ]:


def plot_top_towns(row):
    
    marker_colour = town_colors[row['city']]
    
    folium.Circle(location=[row.latitude, row.longitude],
                  color=marker_colour,
                  tooltip = "<h5 style='text-align:center;font-weight: bold'>Img Name : "+row.filename+"</h5>"+
                            "<hr style='margin:10px;'>"+
                            "<ul style='color: #444;list-style-type:circle;align-item:left;"+
                            "padding-left:20px;padding-right:20px'>"+
                            "<li>Town : "+str(row.city)+"</li>"+
                            "<li>Lat : "+str(row.latitude)+"</li>"+
                            "<li>Long : "+str(row.longitude)+"</li>"+
                            "<li>Altitude : "+str(row.altitude)+"</li>"+
                            "<li>Time date : "+str(row.time_date)+"</li></ul>",
                  radius=20, weight=6).add_to(m)
    
m = folium.Map(location=[top_towns_df['latitude'].mean(), 
                         top_towns_df['longitude'].mean()], 
               tiles='OpenStreetMap',
               min_zoom=7, max_zoom=12, zoom_start=7.5)

# iterate through all rows and plot coords
top_towns_df.apply(plot_top_towns, axis = 1)

m


# ### Processing data into a form suitable for deep learning
# 
# Since all the images were taken in Scotland, it is not feasible to classify images by Country, since they are all the same. However, a potential option is to train a model to classify by County, City or Town. Another option could be to classify location based on location proximity, which could be achieved using a clustering algorithm to assign labels.
# 
# For this work, we'll classify images based on the town the image was taken in. This will be achieved by selected only the top n towns analysed above. Before we can do this, we need to further process our data. This will involve the following steps: 
# 
# 1. Seperate our data according to the classification we want - in this case, we'll seperate the data by the town the image was taken in. This should be simple using the existing meta-data dataframe we created earlier.
# 
# 2. Creation of a new set of directories: training, validation and test. Each of these directories will require n sub-directories for each of the n class labels (towns).
# 
# 3. Selection of splits for training, validation and testing from the original dataset, and movement of these into the applicable new directories created in the previous step.
# 
# 4. Pre-processing our data with the following: conversion from JPEG into RGB pixel grids formatted as floating-point tensors, rescaling and standardisation of our image pixel values so that they lie between 0 and 1.
# 
# The formatting of our data needs to be organised into a specific directory format so that we can apply useful tools such as data augmentation and transformations to our data prior to feeding into a machine learning model.

# In[ ]:


# obtain our data classes (output labels) using top towns from the data
classes = [town.lower() for town in top_towns]

# create our new directories - pathlib Path to avoid preexisting errors
base_dir = os.path.join(os.getcwd(), 'Base_Data')
Path(base_dir).mkdir(parents=True, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# form each of the dirs above
for directory in [train_dir, validation_dir, test_dir]:
    Path(directory).mkdir(parents=True, exist_ok=True)
    
# create sub-directories for each town class for train, val and test dirs
for town_class in classes:
    # create sub-directories within training directory
    current_train_dir = os.path.join(train_dir, town_class)
    Path(current_train_dir).mkdir(parents=True, exist_ok=True)
    
    # repeat for validation dir
    current_val_dir = os.path.join(validation_dir, town_class)
    Path(current_val_dir).mkdir(parents=True, exist_ok=True)
    
    # repeat for test dir
    current_test_dir = os.path.join(test_dir, town_class)
    Path(current_test_dir).mkdir(parents=True, exist_ok=True)


# ###### Form splits of our data using the file paths obtained for each image in the meta-data dataframe previously.

# In[ ]:


# create training, validation and test splits for all images using the file paths
X_path = top_towns_df['filepath'].values
y = top_towns_df['city'].values

# first split - training + validation split combined, and seperate 10% test split.
X_train_val_paths, X_test_paths, y_train, y_test = train_test_split(X_path, y, 
                                                                    test_size=0.1, 
                                                                    random_state=1, 
                                                                    stratify=y)

# second split - 75% training and 25% validation data
X_train_paths, X_val_paths, y_train, y_val = train_test_split(X_train_val_paths, 
                                                              y_train, 
                                                              test_size=0.25, 
                                                              random_state=1, 
                                                              stratify=y_train)


# ###### Plot number of classes within each data split

# In[ ]:


# get counts of class labels within each split
trg_towns, trg_counts =  np.unique(y_train, return_counts=True)
val_towns, val_counts =  np.unique(y_val, return_counts=True)
test_towns, test_counts =  np.unique(y_test, return_counts=True)

# plot number of classes within each data split for confirmation
fig = plt.figure(figsize=(12, 4))
split_types = ['Training', 'Validation', 'Test']

for i, data_split in enumerate([trg_counts, val_counts, test_counts]):
    
    ax = fig.add_subplot(1, 3, i+1)
    sns.barplot(x = trg_towns, y = data_split)
    plt.xticks(rotation=90)
    plt.title(split_types[i])

plt.show()


# We're very limited with test cases for some of our classes (i.e. Aberlour and Sandbank). We're also not very well balanced, with some classes having many more images for training / testing compared to others. Despite these issues, this distribution of data for training, validation and testing will suffice for now.

# ###### Resize and move images from original directory into associated training, validation and test directories

# In[ ]:


get_ipython().system('ls /kaggle/working/Base_Data/train/')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# copy training data\nfor i, img_location in enumerate(X_train_paths):\n    class_label = y_train[i].lower()\n    img_name = f"train_{i}.jpg"\n    src_loc = img_location\n    dest_loc = os.path.join(train_dir, class_label, img_name)\n    \n    # resize img and then move\n    img = Image.open(src_loc)\n    img_new = img.resize((504,378), Image.ANTIALIAS)\n    img_new.save(dest_loc, \'JPEG\', quality=90)\n    \n    # move img without resizing using shutil\n    #_ = shutil.copyfile(src_loc, dest_loc)\n\n# copy validation data\nfor i, img_location in enumerate(X_val_paths):\n    class_label = y_val[i].lower()\n    img_name = f"validation_{i}.jpg"\n    src_loc = img_location\n    dest_loc = os.path.join(validation_dir, class_label, img_name)\n    \n    # resize img and then move\n    img = Image.open(src_loc)\n    img_new = img.resize((504,378), Image.ANTIALIAS)\n    img_new.save(dest_loc, \'JPEG\', quality=90)\n    \n    #_ = shutil.copyfile(src_loc, dest_loc)\n    \n# copy test data\nfor i, img_location in enumerate(X_test_paths):\n    class_label = y_test[i].lower()\n    img_name = f"test_{i}.jpg"\n    src_loc = img_location\n    dest_loc = os.path.join(test_dir, class_label, img_name)\n    \n    # resize img and then move\n    img = Image.open(src_loc)\n    img_new = img.resize((504,378), Image.ANTIALIAS)\n    img_new.save(dest_loc, \'JPEG\', quality=90)\n    \n    #_ = shutil.copyfile(src_loc, dest_loc)')


# ###### Data preprocessing and augmentation of our images using Keras ImageDataGenerator

# In[ ]:


img_height, img_width = 299, 299
batch_size = 10

# training data augmentation - rotate, shear, zoom and flip
train_datagen = ImageDataGenerator(
    rotation_range = 30,
    rescale = 1.0 / 255.0,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip=True)

# no augmentation for test data - only rescale
test_datagen = ImageDataGenerator(rescale = 1. / 255.0)

# generate batches of augmented data from training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# generate val data from val dir
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = len(train_generator.classes)
nb_validation_samples = len(validation_generator.classes)

# create pandas dataframes for our train data
training_data = pd.DataFrame(train_generator.classes, columns=['classes'])
testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])


# ---
# ## 2. Formation of a Deep Convolutional Neural Network
# 
# We need to take our dataset of jpeg images and preprocess them accordingly prior to use in our deep learning model. In summary, we need to perform the following:
# 
# - Read in each of our images as a jpeg file
# - Decode each image into floating-point tensor form, with RGB grids of pixels for each image
# - Standardise our images through rescaling of the pixel values.
# 
# These functions are performed automatically using the data generators we created previously. For our first convolutional neural network, we'll form a custom smaller sized ConvNet.

# In[ ]:


def create_CNN():
    """ Basic CNN with 4 Conv layers, each followed by a max pooling """
    cnn_model = Sequential()
    
    # four Conv layers with max pooling
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    
    # flatten output and feed to dense layer, via dropout layer
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(512, activation='relu'))
    
    # add output layer - softmax with 10 outputs
    cnn_model.add(Dense(10, activation='softmax'))
    
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return cnn_model


# In[ ]:


CNN_model = create_CNN()
CNN_model.summary()


# ###### **Only run the following code once - simply import the training weights and history file to save an hour or training again**

# In[ ]:


#history = CNN_model.fit_generator(train_generator, epochs=30, 
#                                  validation_data=validation_generator, shuffle=True)


# In[ ]:


# save model as a HDF5 file with weights + architecture
#CNN_model.save('Basic_CNN_model_1.h5')

# save the history of training to a datafile for later retrieval
#with open('train_history_basic_CNN_model_1.pickle', 'wb') as pickle_file:
#        pickle.dump(history.history, pickle_file)


# In[ ]:


# if already trained - import history file and training weights
CNN_model = load_model('/kaggle/input/basic-cnn-model/Basic_CNN_model_1.h5')

# get history of trained model
with open('/kaggle/input/basic-cnn-model/train_history_basic_CNN_model_1.pickle', 'rb') as handle:
    history = pickle.load(handle)


# In[ ]:


hist_dict_1 = history

trg_loss = hist_dict_1['loss']
val_loss = hist_dict_1['val_loss']

trg_acc = hist_dict_1['accuracy']
val_acc = hist_dict_1['val_accuracy']

epochs = range(1, len(trg_acc) + 1)

# plot losses and accuracies for training and validation 
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(epochs, trg_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='x', label='Validation Loss')
plt.title("Training / Validation Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
plt.legend(loc='best')

ax = fig.add_subplot(1, 2, 2)
plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')
plt.title("Training / Validation Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
plt.legend(loc='best')
plt.show()


# ###### Evaluate on the test dataset

# In[ ]:


test_generator = test_datagen.flow_from_directory(test_dir, target_size=(299, 299), 
                                                  batch_size=4, class_mode='categorical')

test_loss, test_accuracy = CNN_model.evaluate_generator(test_generator)
print(f"Test accuracy: {test_accuracy}")


# When considering we have 10 different classes, this is not bad for a first attempt at classifying locations based only on visual features of the images! Especially when considering how ambiguous and vague some of the images are - this is an extremely difficult task even for humans that know the locations. 
# 
# Clearly, there is still a lot of scope for improvement of our model as it stands. One potential option could be to perform transfer learning and fine-tuning of an existing high-performing CNN, such as Xception, Inception, VGG19 or ResNet50. This will be formed in the next section.

# ---
# ## 3. Improving our model using transfer learning
# 
# For this we'll obtain a highly trained convolutional base using an existing xception network. We'll take this and train additional layers on top tailored to our image geolocation problem, which should hopefully provide a boost to the performance obtained from the simple ConvNet produced ealier.

# In[ ]:


from keras.applications import xception


# In[ ]:


# create our pretrained convolutonal base from xception
conv_base = xception.Xception(weights='imagenet', include_top=False)


# In[ ]:


conv_base.summary()


# We've only imported the xception network without its top fully-connected layers, since we want to design our own final layers suitable for our geolocation problem. 
# 
# We need to ensure that we have frozen all the existing layers of the Xception model, which is highly trained on the pre-existing imagenet data.

# In[ ]:


for layer in conv_base.layers:
  layer.trainable = False


# Create our xception transfer learning model by adding a small dense network trained on top of the base conv network:

# In[ ]:


tl_xception = Sequential()

# add pre-trained xception base
tl_xception.add(conv_base)

# flatten and add dense layer, with dropout
tl_xception.add(GlobalAveragePooling2D())
tl_xception.add(Dropout(0.5))
tl_xception.add(Dense(256, activation='relu'))

# output softmax, with 10 classes
tl_xception.add(Dense(10, activation='softmax'))

tl_xception.compile(loss='categorical_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])


# Lets set up checkpointing to save the best performance found on the val dataset:

# In[ ]:


# set up a check point for our model - save only the best val performance
save_path ="tl_xception_1_best_weights.hdf5"

trg_checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', 
                                 verbose=1, save_best_only=True, mode='max')

trg_callbacks = [trg_checkpoint]


# In[ ]:


# batch steps before an epoch is considered complete (trg_size / batch_size):
steps_per_epoch = np.ceil(nb_train_samples/batch_size)

# validation batch steps (val_size / batch_size):
val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)


# ###### **Only run the following once on a GPU - after this just import the training weights and history file to avoid having to re-train.**

# In[ ]:


#history = tl_xception.fit(train_generator, epochs=25, 
#                          steps_per_epoch=steps_per_epoch, 
#                          validation_data=validation_generator, 
#                          validation_steps=val_steps_per_epoch,
#                          callbacks=trg_callbacks,
#                          shuffle=True)


# In[ ]:


# save model as a HDF5 file with weights + architecture
#tl_xception.save('tl_xception_1.h5')

# save the history of training to a datafile for later retrieval
#with open('tl_xception_history_1.pickle', 
#          'wb') as pickle_file:
#        pickle.dump(history.history, pickle_file)

loaded_model = False


# If already trained and we have an existing weight and history file, run the following:

# In[ ]:


# if already trained - import history file and training weights
tl_xception = load_model('/kaggle/input/inception-transfer-learning-model/tl_xception_1_model.hdf5')

# get history of trained model
with open('/kaggle/input/inception-transfer-learning-model/tl_xception_history_1.pickle', 'rb') as handle:
    history = pickle.load(handle)
    
loaded_model = True


# In[ ]:


# if loaded model set history accordingly
if loaded_model:
    hist_dict_2 = history
else:
    hist_dict_2 = history.history

trg_loss = hist_dict_2['loss']
val_loss = hist_dict_2['val_loss']

trg_acc = hist_dict_2['accuracy']
val_acc = hist_dict_2['val_accuracy']

epochs = range(1, len(trg_acc) + 1)

# plot losses and accuracies for training and validation 
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(epochs, trg_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='x', label='Validation Loss')
plt.title("Training / Validation Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
plt.legend(loc='best')

ax = fig.add_subplot(1, 2, 2)
plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')
plt.title("Training / Validation Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
plt.legend(loc='best')
plt.show()


# ###### Evaluate on the test dataset

# In[ ]:


test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(299, 299), 
                                                  batch_size=5, 
                                                  class_mode='categorical')

test_loss, test_accuracy = tl_xception.evaluate(test_generator, steps=10)
print(f"Test accuracy: {test_accuracy}")


# This more complex model actually performs worse than the initial ConvNet produced. Its likely that with some further work, research and tinkering of the transfer learning model the performance could be made much better. 
# 
# In addition, we could perform fine-tuning of the actual Convolutional Layers within the Xception (or other chosen base convolutional layer network). This would allow our model to better generalise to the types of images we are using within the geolocation dataset.
# 
# All things considered, we are always going to struggle to effectively classify with this dataset due to the limited number of training samples. Through collection of many more geolocated images with a wider range of examples and locations, I'm sure we could produce a much more impressive model.

# ---
# ## 4. Visualising predictions from the test data with our model(s)
# 
# To decode one-hot encoded predictions back into class labels, we need to create a lookup dictionary from the decoded class labels, as follows:

# In[ ]:


# get class labels dict containing index of each class for decoding predictions
class_labels = train_generator.class_indices

# obtain a reverse dict to convert index into class labels
reverse_class_index = {i : class_label for class_label, i in class_labels.items()}


# In[ ]:


def process_and_predict_img(image_path, model):
    """ Utility function for making predictions for an image. """
    img_path = image_path
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = test_datagen.standardize(x)
    predictions = model.predict(x)
    return img, predictions


# For interest, we can also get back the top *n* predictions to see how certain (or uncertain) or model was in making a given prediction:

# In[ ]:


def top_n_predictions(predict_probs, top_n_labels=3):
    """ Obtain top n prediction indices for array of predictions """
    top_indices = np.argpartition(predict_probs[0], -top_n_labels)[-top_n_labels:]
    
    # negate prediction array to sort in descending order
    sorted_top = top_indices[np.argsort(-predict_probs[0][top_indices])]
    
    # dict comp to create dict of labels and probs
    labels = {"label_" + str(i + 1) : (reverse_class_index[index].capitalize(), 
                                       predict_probs[0][index]) for i, index in enumerate(sorted_top)}
    return labels


# ###### Prediction on a single example from the test set

# In[ ]:


img, prediction = process_and_predict_img(test_dir + '/caol/test_26.jpg', 
                                          model=tl_xception)
top_labels = top_n_predictions(prediction, 
                               top_n_labels=3)
plt.imshow(img)
plt.title(f"Location: {top_labels['label_1'][0]}")
plt.show()

print("Top predictions:")
for label in top_labels:
    print("- {0}: {1:.2f}%".format(top_labels[label][0], top_labels[label][1] * 100))


# ###### Prediction on a range of random examples from the test set with the Xception Transfer Learning Model:

# In[ ]:


example_test_i = np.random.permutation(len(y_test))[:12]
example_test_img = X_test_paths[example_test_i]
example_test_y = y_test[example_test_i]

# create fig to display 12 different predictions
fig = plt.figure(figsize=(15,9))
img_num = 0

for i in range(12):
    ax = fig.add_subplot(3, 4, img_num + 1)
    
    img_path = example_test_img[img_num]
    
    # make prediction on image - select desired model (e.g. CNN_basic, or tl_xception)
    img, predictions = process_and_predict_img(img_path, model=tl_xception)
    top_labels = top_n_predictions(predictions, top_n_labels=3)
    
    prediction_string = ""
    for label in top_labels:
        prediction_string += f"- {top_labels[label][0]}: {top_labels[label][1]*100:.2f}% \n"
    
    ax.imshow(img)
    
    #title = reverse_class_index[np.argmax(predictions,axis=-1)[0]].capitalize()
    ax.set_title(f"Town: {example_test_y[img_num]}")
    ax.set_xlabel(f"Predictions: \n{prediction_string}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    img_num += 1

plt.tight_layout()
plt.show()


# ###### Lets predict on the same examples, but using the original Basic CNN model:

# In[ ]:


# create fig to display 12 different predictions
fig = plt.figure(figsize=(15,9))
img_num = 0

for i in range(12):
    ax = fig.add_subplot(3, 4, img_num + 1)
    
    img_path = example_test_img[img_num]
    
    # make prediction on image - select desired model (e.g. CNN_basic, or tl_xception)
    img, predictions = process_and_predict_img(img_path, model=CNN_model)
    top_labels = top_n_predictions(predictions, top_n_labels=3)
    
    prediction_string = ""
    for label in top_labels:
        prediction_string += f"- {top_labels[label][0]}: {top_labels[label][1]*100:.2f}% \n"
    
    ax.imshow(img)
    
    #title = reverse_class_index[np.argmax(predictions,axis=-1)[0]].capitalize()
    ax.set_title(f"Town: {example_test_y[img_num]}")
    ax.set_xlabel(f"Predictions: \n{prediction_string}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    img_num += 1

plt.tight_layout()
plt.show()


# ---
# 
# ## Finally - remove dataset from Kaggle output directory to prevent thousands of images being output (uncomment code if you'd like to keep / download these)

# In[ ]:


try:
    shutil.rmtree(base_dir)
except OSError as e:
    print("Error: %s : %s" % (base_dir, e.strerror))

