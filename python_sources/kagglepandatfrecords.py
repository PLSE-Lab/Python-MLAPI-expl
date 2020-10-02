#!/usr/bin/env python
# coding: utf-8

# # Getting started with the PANDA dataset
# 
# This notebook shows a few methods to load and display images from the PANDA challenge dataset. The dataset consists of around 11.000 whole-slide images (WSI) of prostate biopsies from Radboud University Medical Center and the Karolinska Institute. 
# 

# In[ ]:


import os

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import skimage

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go

import cv2

import tensorflow as tf

print(tf.__version__)

get_ipython().system('pip install tensorflow-transform')
import tensorflow_transform as tft


# In[ ]:


# Create a dictionary describing the features.
image_mask_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def my_numpy_func(x): 
    # x will be a numpy array with the contents of the input to the 
    # tf.function 
    print("in my_numpy_func")
    return x


def parse_image_function(example_proto1, example_proto2):
    # Parse the input tf.Example proto using the dictionary above.
    #(example_proto1, example_proto2) = example_proto
    image_features = tf.io.parse_example(example_proto1, image_mask_feature_description)
    mask_features = tf.io.parse_example(example_proto2, image_mask_feature_description)
        
    image_raw = image_features['image_raw']
    image_width = image_features['width'] 
    image_height = image_features['height']
    image_depth = image_features['depth']
    
    mask_raw = mask_features['image_raw']
    mask_width = mask_features['width']
    mask_height = mask_features['height']
    mask_depth = mask_features['depth']
       
    data = tf.io.decode_image(image_raw)
    print(data.shape)
    data = tf.image.convert_image_dtype(data, tf.float32)
    data = tf.image.resize(data, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True, antialias=True)
    data = tf.image.pad_to_bounding_box(data, 0, 0, 128, 128)
    
    
    mask_data = tf.io.decode_image(mask_raw)
    print(mask_data.shape)
    #mask_data = tf.image.convert_image_dtype(mask_data, tf.float32)
    mask_data = tf.image.resize(mask_data, (128, 128), preserve_aspect_ratio=True)
    mask_data = tf.image.pad_to_bounding_box(mask_data, 0, 0, 128, 128)

    
    print("displaying image")
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(data)
    
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    plt.subplot(1,3,2)
    plt.imshow(mask_data[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    
    plt.show()
    
    #data = tf.reshape(data, [128,128,3])
    #mask_data = tf.reshape(mask_data, [128,128,3])
    
    return data.numpy(), mask_data.numpy(), data.shape
    #return data, mask_data, data.shape


# In[ ]:


# Location of the training images
tfrecord_dir = '/kaggle/input/tfrecords-panda'

img_test_tf_records = ["images.4.tfrecords"]
img_test_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in img_test_tf_records ]
img_train_tf_records = ["images.2.tfrecords", "images.3.tfrecords"]
img_train_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in img_train_tf_records ]

mask_test_tf_records = ["images_mask.4.tfrecords"]
mask_test_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in mask_test_tf_records ]
mask_train_tf_records = ["images_mask.2.tfrecords", "images_mask.3.tfrecords"]
mask_train_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in mask_train_tf_records ]



image_train_dataset = tf.data.TFRecordDataset(img_train_tf_records)
print(image_train_dataset)
mask_train_dataset = tf.data.TFRecordDataset(mask_train_tf_records)

image_test_dataset = tf.data.TFRecordDataset(img_test_tf_records)
mask_test_dataset = tf.data.TFRecordDataset(mask_test_tf_records)

train_dataset = tf.data.Dataset.zip((image_train_dataset, mask_train_dataset))
print(train_dataset)
test_dataset = tf.data.Dataset.zip((image_test_dataset, mask_test_dataset))

def set_shapes(img, label, img_shape):
    img.shape=img_shape
    label.shape=img_shape
    return img, label


train=train_dataset.map(lambda x1,x2: tf.py_function(func = parse_image_function , inp=[x1,x2], Tout=[tf.float32, tf.float64, tf.int64]))
print(train)

train_x_img=[]
train_x_mask=[]
for image, mask, shape in train.take(2):
    train_x_img.append(image)
    train_x_mask.append(mask)
    
train_x_img = np.array(train_x_img)
train_x_mask = np.array(train_x_mask)
#print(train_x_img.shape)
    
def gen(): 
  for image, mask, shape in train: 
    yield image, np.expand_dims(mask[:,:,0],2)
    
train_x = tf.data.Dataset.from_generator( 
    gen, 
    (tf.int64, tf.int64),
    (tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 1]))) 
print(train_x)


test=test_dataset.map(lambda x1,x2: tf.py_function(func=parse_image_function, inp=[x1,x2], Tout=[tf.float32, tf.float64, tf.int64]))

test_x_img=[]
test_x_mask=[]
for image, mask, shape in test.take(2):
    test_x_img.append(image)
    test_x_mask.append(mask)

test_x_img = np.array(test_x_img)
test_x_mask = np.array(test_x_mask)
#print(test_x_img.shape)

    
def gen(): 
  for image, mask, shape in test: 
    print(image.shape, mask.shape)
    yield image, np.expand_dims(mask[:,:,0],2)
test_x = tf.data.Dataset.from_generator( 
     gen, 
     (tf.int64, tf.int64),
     (tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 1]))) 

print(test_x)
#test=test.map(lambda img, mask, shape:tf.numpy_function(func = set_shapes, inp = [img, mask, shape], Tout=[tf.float32, tf.float64]))
#test=test_dataset.map(parse_image_function)
#for image, mask in test:
#    print("hereee")

#img_shape = images[0].shape  # images is a list of numpy.ndarray
#ds = ds.map(lambda img, label: tf.py_function( ... ) )
#ds = ds.map(lambda img, label: set_shapes(img, label, img_shape) )



# In[ ]:


get_ipython().system('pip install -q git+https://github.com/tensorflow/examples.git')
from tensorflow_examples.models.pix2pix import pix2pix

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# In[ ]:


OUTPUT_CHANNELS = 6
BATCH_SIZE = 4
train_dataset = train_x.cache().shuffle(100).batch(4, drop_remainder=True).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_x.batch(4)
print(train_dataset)
print(test_dataset)


model = unet_model(OUTPUT_CHANNELS)

for data, label in test_x.take(1):
    pred_mask = model.predict(data[tf.newaxis, ...])
    pred_mask = create_mask(pred_mask)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask))
    #plt.imshow(pred_mask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    plt.show()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#tf.keras.utils.plot_model(model, show_shapes=True)

#print(train_x_mask.shape)

model_history = model.fit(train_dataset, epochs=1,
                          steps_per_epoch=2,
                          validation_steps=2,
                          validation_data=test_dataset)

#model_history = model.fit(train_x_img, train_x_mask[:,:,:,0], epochs=1,
#                          steps_per_epoch=5,
#                          validation_steps=2,
#                          validation_data=(test_x_img, test_x_mask[:,:,:,0]))


# In[ ]:


# Location of the training images
tfrecord_dir = '/kaggle/input/tfrecords-panda'


img_test_tf_records = ["images.4.tfrecords"]
img_test_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in img_test_tf_records ]
img_train_tf_records = ["images.2.tfrecords", "images.3.tfrecords"]
img_train_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in img_train_tf_records ]

mask_test_tf_records = ["images_mask.4.tfrecords"]
mask_test_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in mask_test_tf_records ]
mask_train_tf_records = ["images_mask.2.tfrecords", "images_mask.3.tfrecords"]
mask_train_tf_records = [os.path.join(tfrecord_dir, tf_record) for tf_record in mask_train_tf_records ]



image_test_dataset = tf.data.TFRecordDataset(img_test_tf_records)
mask_test_dataset = tf.data.TFRecordDataset(mask_test_tf_records)
    
image_train_dataset = tf.data.TFRecordDataset(img_train_tf_records)
mask_train_dataset = tf.data.TFRecordDataset(mask_train_tf_records)



def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
   return tf.io.parse_single_example(example_proto, image_mask_feature_description)


parsed_image_test_dataset = image_test_dataset.map(_parse_image_function)
parsed_mask_test_dataset = mask_test_dataset.map(_parse_image_function)
print(parsed_image_test_dataset)

parsed_image_train_dataset = image_train_dataset.map(_parse_image_function)
parsed_mask_train_dataset = mask_train_dataset.map(_parse_image_function)
print(parsed_image_train_dataset)



for image_features, mask_features in zip(parsed_image_train_dataset.take(2), parsed_mask_train_dataset.take(2)):
    image_raw = image_features['image_raw']
    image_width = image_features['width'].numpy()
    image_height = image_features['height'].numpy()
    image_depth = image_features['depth'].numpy()
    
    mask_raw = mask_features['image_raw']
    mask_width = mask_features['width'].numpy()
    mask_height = mask_features['height'].numpy()
    mask_depth = mask_features['depth'].numpy()
       
    data = tf.io.decode_image(image_raw)
    data = tf.image.resize(data, (1280, 1280), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True, antialias=True)
    data = tf.image.pad_to_bounding_box(data, 0, 0, 1280, 1280)
    #img = img - tft.mean(img)
    print(data.shape, image_width*image_height*image_depth)
    mask_data = tf.io.decode_image(mask_raw)
    mask_data = tf.image.resize(mask_data, (1280, 1280), preserve_aspect_ratio=True)
    mask_data = tf.image.pad_to_bounding_box(mask_data, 0, 0, 1280, 1280)
    print(mask_data.shape, mask_width*mask_height*mask_depth)
    
    #img = cv2.imdecode(data.numpy(), cv2.IMREAD_UNCHANGED)
    print(data.shape)
    #mask_img = cv2.imdecode(mask_data.numpy(), cv2.IMREAD_UNCHANGED)
    print(mask_data.shape)
    print("displaying image")
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(data)
    #plt.show()
    
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    plt.subplot(1,3,2)
    plt.imshow(mask_data[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    #plt.show()
    
    pred_mask = model.predict(data[tf.newaxis, ...])
    pred_mask = create_mask(pred_mask)
    plt.subplot(1,3,3)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask))
    #plt.imshow(pred_mask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    plt.show()
    


# In[ ]:


record_file = 'images_mask.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:

    i = 0
    for index, row in train_labels.iterrows():
        mask_file = os.path.join(mask_dir, f'{index}_mask.tiff')
        file = os.path.join(data_dir, f'{index}.tiff')
        print(i, mask_file, row["data_provider"])  
        i=i+1
        if i<=150:
            continue
        if i>180:
            break;

        if os.path.isfile(mask_file):
            biopsy = skimage.io.MultiImage(mask_file)
            skimage.io.imsave(f'./{index}_mask.bmp', biopsy[1])

            #image_raw = tf.io.read_file(f'./{index}.bmp')
            image_raw = open(f'./{index}_mask.bmp', 'rb').read()
            filename = f'./{index}_mask.bmp'
            get_ipython().system(' rm {filename}')
            image_decoded = tf.image.decode_image(image_raw)
            image_shape = image_decoded.shape



            feature = {
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[row["isup_grade"]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))     
            writer.write(tf_example.SerializeToString())   


# In[ ]:


record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:

    i = 0
    for index, row in train_labels.iterrows():
        mask_file = os.path.join(mask_dir, f'{index}_mask.tiff')
        file = os.path.join(data_dir, f'{index}.tiff')
        print(i, mask_file, row["data_provider"])  
        i=i+1
        if i<=150:
            continue
        if i>180:
            break;

        if os.path.isfile(file):
            biopsy = skimage.io.MultiImage(file)
            skimage.io.imsave(f'./{index}.bmp', biopsy[1])

            #image_raw = tf.io.read_file(f'./{index}.bmp')
            image_raw = open(f'./{index}.bmp', 'rb').read()
            filename = f'./{index}.bmp'
            get_ipython().system(' rm {filename}')
            image_decoded = tf.image.decode_image(image_raw)
            image_shape = image_decoded.shape



            feature = {
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[row["isup_grade"]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))     
            writer.write(tf_example.SerializeToString())   
            


# In[ ]:


# Location of the training images
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'
output_dir = '/kaggle/working'

# Location of training labels
train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')


# # Quickstart: reading a patch in 4 lines
# 
# The example below shows in 4 lines how to extract a patch from one of the slides using OpenSlide.

# In[ ]:


# Open the image (does not yet read the image into memory)
image = openslide.OpenSlide(os.path.join(data_dir, '005e66f06bce9c2e49142536caf2f6ee.tiff'))

# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = image.read_region((17800,19500), 0, (256, 256))

# Display the image
display(patch)

# Close the opened slide after use
image.close()


# # Using OpenSlide to load the data
# 
# In the following sections we will load data from the slides with [OpenSlide](https://openslide.org/api/python/). The benefit of OpenSlide is that we can load arbitrary regions of the slide, without loading the whole image in memory. Want to interactively view a slide? We have added an [interactive viewer](#Interactive-viewer-for-slides) to this notebook in the last section.
# 
# You can read more about the OpenSlide python bindings in the documentation: https://openslide.org/api/python/
# 
# ## Loading a slide
# 
# Before we can load data from a slide, we need to open it. After a file is open we can retrieve data from it at arbitratry positions and levels.
# 
# ```python
# biopsy = openslide.OpenSlide(path)
# # do someting with the slide here
# biopsy.close()
# ```

# For this tutorial, we created a small function to show some basic information about a slide. Additionally, this function display a small thumbnail of the slide. All images in the dataset contain this metadata and you can use this in your data pipeline.

# In[ ]:


def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    """Print some basic information about a slide"""
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


# Running the cell below loads four example biopsies using OpenSlide. Some things you can notice:
# 
# - The image dimensions are quite large (typically between 5.000 and 40.000 pixels in both x and y).
# - Each slide has 3 levels you can load, corresponding to a downsampling of 1, 4 and 16. Intermediate levels can be created by downsampling a higher resolution level.
# - The dimensions of each level differ based on the dimensions of the original image.
# - Biopsies can be in different rotations. This rotation has no clinical value, and is only dependent on how the biopsy was collected in the lab.
# - There are noticable color differences between the biopsies, this is very common within pathology and is caused by different laboratory procedures.
# 

# In[ ]:


example_slides = [
    '005e66f06bce9c2e49142536caf2f6ee',
    '00928370e2dfeb8a507667ef1d4efcbb',
    '007433133235efc27a39f11df6940829',
    '024ed1244a6d817358cedaea3783bbde',
]

for case_id in example_slides:
    biopsy = openslide.OpenSlide(os.path.join(data_dir, f'{case_id}.tiff'))
    print_slide_details(biopsy)
    biopsy.close()
    
    # Print the case-level label
    print(f"ISUP grade: {train_labels.loc[case_id, 'isup_grade']}")
    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")


# ## Loading image regions/patches
# 
# With OpenSlide we can easily extract patches from the slide from arbitrary locations. Loading a specific region is done using the [read_region](https://openslide.org/api/python/#openslide.OpenSlide.read_region) function.
# 
# After opening the slide we can, for example, load a 512x512 patch from the lowest level (level 0) at a specific coordinate.
# 

# In[ ]:


biopsy = openslide.OpenSlide(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)


# Using the `level` argument we can easily load in data from any level that is present in the slide. Coordinates passed to `read_region` are always relative to level 0 (the highest resolution).

# In[ ]:


x = 5140
y = 21000
level = 1
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)


# In[ ]:


biopsy.close()


# ## Loading label masks
# 
# Apart from the slide-level label (present in the csv file), almost all slides in the training set have an associated mask with additional label information. These masks directly indicate which parts of the tissue are healthy and which are cancerous. The information in the masks differ from the two centers:
# 
# - **Radboudumc**: Prostate glands are individually labelled. Valid values are:
#   - 0: background (non tissue) or unknown
#   - 1: stroma (connective tissue, non-epithelium tissue)
#   - 2: healthy (benign) epithelium
#   - 3: cancerous epithelium (Gleason 3)
#   - 4: cancerous epithelium (Gleason 4)
#   - 5: cancerous epithelium (Gleason 5)
# - **Karolinska**: Regions are labelled. Valid values:
#   - 0: background (non tissue) or unknown
#   - 1: benign tissue (stroma and epithelium combined)
#   - 2: cancerous tissue (stroma and epithelium combined)
# 
# The label masks of Radboudumc were semi-automatically generated by several deep learning algorithms, contain noise, and can be considered as weakly-supervised labels. The label masks of Karolinska were semi-autotomatically generated based on annotations by a pathologist.
# 
# The label masks are stored in an RGB format so that they can be easily opened by image readers. The label information is stored in the red (R) channel, the other channels are set to zero and can be ignored. As with the slides itself, the label masks can be opened using OpenSlide.

# ### Visualizing the masks (using PIL)
# 
# Using a small helper function we can display some basic information about a mask. To more easily inspect the masks, we map the int labels to RGB colors using a color palette. If you prefer something like `matplotlib` you can also use `plt.imshow()` to directly show a mask (without converting it to an RGB image).

# In[ ]:


def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):
    """Print some basic information about a slide"""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        display(mask_data)

    # Compute microns per pixel (openslide gives resolution in centimeters)
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


# The cells below shows two example masks from the dataset. The first mask is from Radboudumc and shows two different grades of cancer (shown in yellow and orange). The second mask is from Karolinska, the region that contains cancer is higlighted in red.
# 
# Note that, eventhough a biopsy contains cancer, not all epithelial tissue has to be cancerous. Biopsies can contain a mix of cancerous and healthy tissue.

# In[ ]:


mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
print_mask_details(mask, center='radboud')
mask.close()


# In[ ]:


mask = openslide.OpenSlide(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
print_mask_details(mask, center='karolinska')
mask.close()


# ### Visualizing masks (using matplotlib)
# 
# Given that the masks are just integer matrices, you can also use other packages to display the masks. For example, using matplotlib and a custom color map we can quickly visualize the different cancer regions:

# In[ ]:


mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

plt.figure()
plt.title("Mask with default cmap")
plt.imshow(np.asarray(mask_data)[:,:,0], interpolation='nearest')
plt.show()

plt.figure()
plt.title("Mask with custom cmap")
# Optional: create a custom color map
cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
plt.imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
plt.show()

mask.close()


# ### Overlaying masks on the slides
# 
# As the masks have the same dimension as the slides, we can overlay the masks on the tissue to directly see which areas are cancerous. This overlay can help you identifying the different growth patterns. To do this, we load both the mask and the biopsy and merge them using PIL.
# 
# **Tip:** Want to view the slides in a more interactive way? Using a WSI viewer you can interactively view the slides. Examples of open source viewers that can open the PANDA dataset are [ASAP](https://github.com/computationalpathologygroup/ASAP) and [QuPath](https://qupath.github.io/). ASAP can also overlay the masks on top of the images using the "Overlay" functionality. If you use Qupath, and the images do not load, try changing the file extension to `.vtif`.

# In[ ]:


def overlay_mask_on_slide(slide, mask, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Load data from the highest level
    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    # Mask data is present in the R channel
    mask_data = mask_data.split()[0]

    # Create alpha mask
    alpha_int = int(round(255*alpha))
    if center == 'radboud':
        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
    elif center == 'karolinska':
        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)
    
    alpha_content = PIL.Image.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)
    
    if center == 'radboud':
        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
    elif center == 'karolinska':
        # Mapping: {0: background, 1: benign, 2: cancer}
        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)
    
    mask_data.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data.convert(mode='RGB')

    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
    overlayed_image.thumbnail(size=max_size, resample=0)

    display(overlayed_image)


# > Note: In the example below you can also observe a few pen markings on the slide (dark green smudges). These markings are not part of the tissue but were made by the pathologists who originally checked this case. These pen markings are available on some slides in the training set.

# In[ ]:


slide = openslide.OpenSlide(os.path.join(data_dir, '08ab45297bfe652cc0397f4b37719ba1.tiff'))
mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
overlay_mask_on_slide(slide, mask, center='radboud')
slide.close()
mask.close()


# In[ ]:


slide = openslide.OpenSlide(os.path.join(data_dir, '090a77c517a7a2caa23e443a77a78bc7.tiff'))
mask = openslide.OpenSlide(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
overlay_mask_on_slide(slide, mask, center='karolinska', alpha=0.6)
slide.close()
mask.close()


# # Using scikit-image & tifffile to load the data
# 
# As an alternative to OpenSlide, the slides in the PANDA dataset can also be loaded using [scikit-image](https://scikit-image.org/) with [tifffile](https://pypi.org/project/tifffile/) as the backend.
# 
# > **Note:** scikit-image (<= 0.16.x) uses an internal version of the tif loader if the tifffile packages is not installed. This internal version does not support JPEG compression and can not be used to load the images in the dataset. Make sure tifffile is installed before running the examples below. This requirement is already met when running this code in a notebook on Kaggle.
# 
# ## Loading a slide
# 
# Loading a slide with [scikit-image](https://scikit-image.org/) is similar to loading slides with OpenSlide. The major difference between scikit-image and OpenSlide is that scikit-image loads the image into memory. To extract a certain region of the image, you will need to load the whole image at one of the levels.
# 
# The images in the PANDA dataset are relatively small because each biopsy was individually extracted from the source slide. The small size makes it possible to load the slides directly into memory. Still, upon loading the image is uncompressed resulting in larger memory usage.
# 
# Slides are loaded using the [MultiImage](https://scikit-image.org/docs/0.16.x/api/skimage.io.html?highlight=multiimage#skimage.io.MultiImage) class; this class gives the ability to access the individual levels of the image. By default, MultiImage tries to conserve memory usage by only caching the last image level that was accessed.
# 

# In[ ]:


biopsy = skimage.io.MultiImage(os.path.join(data_dir, '0b373388b189bee3ef6e320b841264dd.tiff'))


# The code below loads each individual level. You can check the memory usage of the kernel to see that loading the lowest level can require a considerate amount of memory.

# In[ ]:


for i,level in enumerate(biopsy):
    print(f"Biopsy level {i} dimensions: {level.shape}")
    print(f"Biopsy level {i} memory size: {level.nbytes / 1024**2:.1f}mb")


# In[ ]:


display(PIL.Image.fromarray(biopsy[-1]))


# In[ ]:


# Deleting the object frees up memory
del biopsy


# If you are only interested in the lowest level (highest magnification), you can also load level 0 using [imread](https://scikit-image.org/docs/0.16.x/api/skimage.io.html?highlight=imread#skimage.io.imread):

# In[ ]:


biopsy_level_0 = skimage.io.imread(os.path.join(data_dir, '0b373388b189bee3ef6e320b841264dd.tiff'))
print(biopsy_level_0.shape)
del biopsy_level_0


# ## Loading image regions
# 
# Similar to OpenSlide, we can extract regions from the whole image. Because the image is already in memory, this boils down to a slice on the numpy array. To illustrate we use the same coordinates as in the OpenSlide example:

# In[ ]:


biopsy = skimage.io.MultiImage(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

patch = biopsy[0][y:y+width, x:x+height]

# You can also visualize patches with matplotlib
plt.figure()
plt.imshow(patch)
plt.show()


# To load the same region from level 1, we have to devide the coordinates with the downsample factor (4 per level). This is different from Openslide that always works with coordinates from level 0.

# In[ ]:


x = 5150 // 4
y = 21000 // 4
width = 512
height = 512

patch = biopsy[1][y:y+width, x:x+height]

plt.figure()
plt.imshow(patch)
plt.show()


# In[ ]:


x = 5150 // (4*4)
y = 21000 // (4*4)
width = 512
height = 512

patch = biopsy[2][y:y+width, x:x+height]

plt.figure()
plt.imshow(patch)
plt.show()


# In[ ]:


# Free up memory
del biopsy


# ## Loading label masks
# 
# Loading label masks using scikit-image is similar to loading the slides. As the label information is in the R channel, other channels can be discarded. Please refer to the "OpenSlide - Loading label masks" section for more information about the contents of the label masks.

# In[ ]:


maskfile = skimage.io.MultiImage(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
mask_level_2 = maskfile[-1][:,:,0]

plt.figure()
plt.imshow(mask_level_2)
plt.colorbar()
plt.show()


# In[ ]:


del maskfile


# # Interactive viewer for slides
# 
# Using [Plotly](https://kite.com/python/docs/plotly.graph_objs) we can make an interactive viewer that works inside a notebook. Using this viewer you can load any image from the PANDA dataset and interactively zoom in to specific regions. This viewer is a great way of inspecting the data in more detail.
# 
# > **Note:** The code below only works when you run this notebook yourself. The output is not shown when purely viewing the notebook as it requires access to the source image.
# 
# Want to investigate slides locally on your machine? Using a WSI viewer you can interactively view the slides on your own machine. Examples of open source viewers that can open the PANDA dataset are [ASAP](https://github.com/computationalpathologygroup/ASAP) and [QuPath](https://qupath.github.io/). ASAP can also overlay the masks on top of the images using the "Overlay" functionality. If you use Qupath, and the images do not load, try changing the file extension to `.vtif`.
# 
# 

# In[ ]:


class WSIViewer(object):
    def __init__(self, plot_size = 1000):
        self._plot_size = plot_size
        
    def set_slide(self, slide_path):      
        self._slide = openslide.open_slide(slide_path)
        self._base_dims = self._slide.level_dimensions[-1]
        self._base_ds = self._slide.level_downsamples[-1]
        img_arr = self._slide.read_region((0,0), len(self._slide.level_dimensions[-1]), (self._base_dims[0], self._base_dims[1]))
        
        self._fig = go.FigureWidget(data=[{'x': [0, self._base_dims[0]], 
                                           'y': [0, self._base_dims[1]], 
                                           'mode': 'markers',
                                           'marker': {'opacity': 0}}], # invisible trace to init axes and to support autoresize
                                    layout={'width': self._plot_size, 'height': self._plot_size, 'yaxis' : dict(scaleanchor = "x", scaleratio = 1)})  
        # Set background image
        self._fig.layout.images = [go.layout.Image(
            source = img_arr,  # plotly now performs auto conversion of PIL image to png data URI
            xref = "x",
            yref = "y",
            x = 0,
            y = 0,
            sizex = self._base_dims[0],
            sizey = self._base_dims[1],
            sizing = "stretch",
            layer = "below")]
        self._fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False);        
        self._fig.layout.on_change(self._update_image, 'xaxis.range', 'yaxis.range', 'width', 'height')          

    def _gen_zoomed_image(self, x_range, y_range):
        # Below is a workaround which rounds image requests to multiples of 4, once the libpixman fix is in place these can be removed
        #xstart = x_range[0] * self._base_ds
        #ystart = (self._base_dims[1] - y_range[1]) * self._base_ds 
        xstart = 4 * round(x_range[0] * self._base_ds / 4)
        ystart = 4 * round((self._base_dims[1] - y_range[1]) * self._base_ds / 4)
        xsize0 = (x_range[1] - x_range[0]) * self._base_ds
        ysize0 = (y_range[1] - y_range[0]) * self._base_ds
        if (xsize0 > ysize0):
            req_downs = xsize0 / self._plot_size
        else:
            req_downs = ysize0 / self._plot_size
        req_level = self._slide.get_best_level_for_downsample(req_downs)
        level_downs = self._slide.level_downsamples[req_level]
        # Nasty workaround for buggy container
        level_size_x = int(xsize0 / level_downs)
        level_size_y = int(ysize0 / level_downs)
        new_img = self._slide.read_region((int(xstart), int(ystart)), req_level, (level_size_x, level_size_y)).resize((1000,1000)) # Letting PIL do the resize is faster than plotly
        return new_img
    
    def _update_image(self, layout, x_range, y_range, plot_width, plot_height):
        img = self._fig.layout.images[0]
        # Update with batch_update so all updates happen simultaneously
        with self._fig.batch_update():
            new_img = self._gen_zoomed_image(x_range, y_range)
            img.x = x_range[0]
            img.y = y_range[1]
            img.sizex = x_range[1] - x_range[0]
            img.sizey = y_range[1] - y_range[0]
            img.source = new_img

    def show(self):
        return self._fig


# In[ ]:


viewer = WSIViewer()
viewer.set_slide(os.path.join(data_dir, '08ab45297bfe652cc0397f4b37719ba1.tiff'))
viewer.show()


# In[ ]:




