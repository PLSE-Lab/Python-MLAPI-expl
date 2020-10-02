#!/usr/bin/env python
# coding: utf-8

# ### **Problem Statement:**
# 
# Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer, but if caught early, most melanomas can be cured with minor surgery. In this competition, the task is to use images of skin lesions within the same patient and determine which are likely to represent a melanoma.
# 
# ### **Dataset:**
# Data is provided in the following formats:
# 1. DICOM (Digital Imaging and Communications in Medicine) contains both image and metadata
# 2. JPEG (Images in original size)
# 3. TFRecord (Images resized to 1024x1024)
# 4. Metadata in CSV
# 
# ### **Contents:**
# <b><a href="#one">1. Reading data</a><br></b>
# &emsp;&emsp;<a href="#one.one">1.1. DICOM format</a><br>
# &emsp;&emsp;<a href="#one.two">1.2. JPEG format</a><br>
# &emsp;&emsp;<a href="#one.three">1.3. TFRecord format</a><br>
# <b><a href="#two">2. Quality comparison</a><br></b>
# &emsp;&emsp;<a href="#two.one">2.1. DICOM vs JPEG</a><br>
# &emsp;&emsp;<a href="#two.two">2.2. Source of TFRecord</a><br>
# <b><a href="#three">3. DICOM to PNG/JPEG</a><br></b>
# <b><a href="#four">4. Conclusions</a><br></b>
# <b><a href="#five">5. Next steps</a><br></b>
# 
# ### **Motivation:**
# 1. Learning how to handle of dicom files.
# 2. Compare the quality of the images between the formats provided. 
# 3. Create dataset with the images of highest available quality resized to 512x512, 256x256 etc.,
# 
# This notebook is inspired by this [discussion](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155579#875036) and this [post](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155415).
# 
# ### **Credits:**
# 1. Thanks [@cdeotte](https://www.kaggle.com/cdeotte) for this [notebook](https://www.kaggle.com/cdeotte/how-to-create-tfrecords/notebook) on working with TFRecords.
# 2. Thanks [@Abhishek](https://www.kaggle.com/abhishek) for this [notebook](https://www.kaggle.com/abhishek/convert-to-png-on-steroids-with-actual-data/notebook) on converting images from DICOM to PNG in parallel.

# In[ ]:


import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

from PIL import Image
from skimage import io
import cv2

import tensorflow as tf

import albumentations 

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 15

from pathlib import Path

INPUT_PATH = Path("/kaggle/input/siim-isic-melanoma-classification")


# **<h1 id="one" >1. Reading data </h1>**

# In[ ]:


train_metadata = pd.read_csv(INPUT_PATH/"train.csv");print(f"Train shape: {train_metadata.shape}")
train_metadata.head()


# In[ ]:


# Select a random image for analysis
image_name = np.random.choice(train_metadata.image_name)
image_name


# **<h2 id="one.one" >1.1. DICOM format </h2>**

# In[ ]:


print(f"Reading {image_name} DICOM file")
ds = pydicom.dcmread(INPUT_PATH/f"train/{image_name}.dcm")
print(f"Contents of {image_name} DICOM file")
ds


# * DICOM file processing is discussed [here](https://www.kaggle.com/avirdee/fastai2-dicom-starter#Fastai2-DICOM-starter) and more details on the format are available [here].(http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.4)
# * Key part in order to process the image for this competition is **(0028, 0004) Photometric Interpretation**.
# * I have verified that all the DICOM files of train and test have images in **YBR_FULL_422** color scale. 

# In[ ]:


dicom_arr = convert_color_space(ds.pixel_array, "YBR_FULL_422", "RGB")
dicom_img = Image.fromarray(dicom_arr)


# **<h2 id="one.two" >1.2. JPEG format </h2>**

# In[ ]:


jpeg_img = Image.open(f"{INPUT_PATH}/jpeg/train/{image_name}.jpg")
jpeg_arr = np.asarray(jpeg_img)


# **<h2 id="one.three" >1.3. TFRecord format </h2>**

# In[ ]:


def mapping_func(serialized_example):
    TFREC_FORMAT = {
                    "image_name": tf.io.FixedLenFeature([], tf.string),
                    "image": tf.io.FixedLenFeature([], tf.string)
                    }
    example = tf.io.parse_single_example(serialized_example, features=TFREC_FORMAT)
    return example

def filter_tfrecords(dataset, image_name):
    record_dataset = dataset.filter(lambda example: tf.equal(example["image_name"], image_name))
    example = next(iter(record_dataset))
    arr = tf.image.decode_jpeg(example['image'], channels=3).numpy()
    return arr


# In[ ]:


dataset = tf.data.TFRecordDataset(tf.io.gfile.glob('/kaggle/input/siim-isic-melanoma-classification/tfrecords/train*.tfrec'))
dataset = dataset.map(mapping_func)

tfrecord_arr = filter_tfrecords(dataset, image_name)
tfrecord_img = Image.fromarray(tfrecord_arr)


# In[ ]:


print(f"DICOM array shape: {dicom_arr.shape}")
print(f"JPEG array shape: {jpeg_arr.shape}")
print(f"TFRecord array shape: {tfrecord_arr.shape}")


# Let's see how they look!

# In[ ]:


plt.figure(figsize=(20, 8))
plt.subplot(131), io.imshow(dicom_arr)
plt.title('DICOM Format') 

plt.subplot(132), io.imshow(jpeg_arr)
plt.title('JPEG Format') 

plt.subplot(133), io.imshow(tfrecord_arr)
plt.title('TFRecord Format') 

plt.show()


# ### **Observations:**
# 
# 1. Both the image from DICOM file and JPEG are of same size.
# 2. Image from TFRecord seems to be cropped and resized to uniform dimensions of 1024x1024 as mentioned in the [data description](https://www.kaggle.com/c/siim-isic-melanoma-classification/data).

# **<h1 id="two" >2. Quality comparison </h1>**

# The differences between the images are hard to be noticed by a human eye. I am exploring the following approaches to compare two images:
# 1. Exact match
# 2. Mean squared error
# 
# Feel free to suggest other approaches to compare two images with respect to the quality. I'd be happy to learn and add it to this analysis

# In[ ]:


is_same_image = lambda a, b: np.allclose(a, b)
mse = lambda a, b: np.square(np.subtract(a, b)).mean()


# **<h2 id="two.one" >2.1. DICOM vs JPEG </h2>**

# In[ ]:


is_same_image(dicom_arr, jpeg_arr)


# In[ ]:


mse(dicom_arr, jpeg_arr) #Mean Squared Error


# * Although DICOM image and JPEG image are of same size, they are not one and the same.
# * MSE only shows if two images are different and more analysis is needed in order to understand their differences in quality.

# In[ ]:


# Write DICOM image to JPEG at 75% quality
dicom_img.save(f"dicom2jpeg_75q_{image_name}.jpg", "JPEG", quality=75)

# Read the image
dicom2jpeg_75q_img = Image.open(f"dicom2jpeg_75q_{image_name}.jpg")
dicom2jpeg_75q_arr = np.asarray(dicom2jpeg_75q_img)


# In[ ]:


is_same_image(jpeg_arr, dicom2jpeg_75q_arr)


# In[ ]:


mse(jpeg_arr, dicom2jpeg_75q_arr)


# ## That's interesting! Let's check this for randomly selected 100 images.

# In[ ]:


def compare(image_name):
    ds = pydicom.dcmread(INPUT_PATH/f"train/{image_name}.dcm")
    dicom_arr = convert_color_space(ds.pixel_array, "YBR_FULL_422", "RGB")
    dicom_img = Image.fromarray(dicom_arr)
    
    jpeg_img = Image.open(f"{INPUT_PATH}/jpeg/train/{image_name}.jpg")
    jpeg_arr = np.asarray(jpeg_img)
    
    # Write DICOM image to JPEG at 75% quality
    dicom_img.save(f"dicom2jpeg_75q_{image_name}.jpg", "JPEG", quality=75)
    # Read the image
    dicom2jpeg_75q_img = Image.open(f"dicom2jpeg_75q_{image_name}.jpg")
    dicom2jpeg_75q_arr = np.asarray(dicom2jpeg_75q_img)
    Path(f"dicom2jpeg_75q_{image_name}.jpg").unlink()
    
    comparison = {
        "image_name": image_name,
        "dicom_vs_jpeg_is_same" : is_same_image(dicom_arr, jpeg_arr),
        "dicom_vs_jpeg_mse": mse(dicom_arr, jpeg_arr),
        "jpeg_vs_dicom2jpeg_75q_is_same": is_same_image(jpeg_arr, dicom2jpeg_75q_arr),
        "jpeg_vs_dicom2jpeg_75q_mse": mse(jpeg_arr, dicom2jpeg_75q_arr)
    }
    return comparison


# In[ ]:


image_names = np.random.choice(train_metadata.image_name, size=100, replace=False)
comparisons = Parallel(n_jobs=8, backend='threading')(delayed(
    compare)(image_name) for image_name in tqdm(image_names, total=len(image_names)))


# In[ ]:


comparisons_df = pd.DataFrame(comparisons)
comparisons_df.head()


# In[ ]:


print(np.sum(~comparisons_df["jpeg_vs_dicom2jpeg_75q_is_same"]))


# In[ ]:


print(np.sum(comparisons_df["jpeg_vs_dicom2jpeg_75q_mse"]))


# ## Wow! JPEG images seem be saved at 75% quality compared to DICOM images

# **<h2 id="two.two" >2.2. Source of TFRecord </h2>**

# In[ ]:


resize_and_crop = albumentations.Compose([
    albumentations.SmallestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
    albumentations.CenterCrop(1024, 1024)
])
resized_cropped_dicom_arr = resize_and_crop(image=dicom_arr)["image"]


# In[ ]:


plt.figure(figsize=(20, 45))

plt.subplot(121), io.imshow(tfrecord_arr)
plt.title('TFRecord Format') 

plt.subplot(122), io.imshow(resized_cropped_dicom_arr)
plt.title('Resized and Cropped DICOM Format') 

plt.show()


# In[ ]:


is_same_image(tfrecord_arr, resized_cropped_dicom_arr)


# In[ ]:


mse(tfrecord_arr, resized_cropped_dicom_arr)


# The images in TFRecord are still different after resizing holding the aspect ratio same as original image. Following are the possible reasons:
# * The images in TFRecord are processed a bit more than just resize and crop.
# * There might be slight loss of information while encoding the image in TFRecord. 
# 
# Note: I am not an expert in Computer Vision and I might very well be wrong here. I look forward to comments/suggestions to experiment and learn.

# **<h1 id="three" >3. DICOM to PNG/JPEG </h1>**

# In[ ]:


# Write DICOM image to JPEG at 100% quality
dicom_img.save(f"dicom2jpeg_100q_{image_name}.jpg", "JPEG", quality=100)

# Read the image
dicom2jpeg_100q_img = Image.open(f"dicom2jpeg_100q_{image_name}.jpg")
dicom2jpeg_100q_arr = np.asarray(dicom2jpeg_100q_img)


# In[ ]:


is_same_image(dicom_arr, dicom2jpeg_100q_arr)


# In[ ]:


mse(dicom_arr, dicom2jpeg_100q_arr)


# In[ ]:


# Write DICOM image to PNG at 100% quality
dicom_img.save(f"dicom2png_100q_{image_name}.png", "PNG", quality=100)

# Read the image
dicom2png_100q_img = Image.open(f"dicom2png_100q_{image_name}.png")
dicom2png_100q_arr = np.asarray(dicom2png_100q_img)


# In[ ]:


is_same_image(dicom_arr, dicom2png_100q_arr)


# In[ ]:


mse(dicom_arr, dicom2png_100q_arr)


# **<h1 id="four" >4. Conclusions </h1>**
# * JPEG images are at 75% quality compared to images in DICOM files
# * There is a minimal loss of information while converting DICOM image to JPEG with 100 quality.
# * There is zero loss of information while converting DICOM image to PNG with 100 quality.

# **<h1 id="five" >5. Next steps </h1>**
# * Explore how important is maintaining aspect ratio while resizing the images to different dimensions (512x512, 1024x1024 etc.,).
# * Use images from DICOM files to resize and convert in to PNG format for modelling.
# * Build models at different qualities and different sizes to understand the performance.
