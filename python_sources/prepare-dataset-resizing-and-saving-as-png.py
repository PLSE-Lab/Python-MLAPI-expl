#!/usr/bin/env python
# coding: utf-8

# Dataset URL: https://www.kaggle.com/guiferviz/rsna_stage1_png_128
# 
# # Preparing dataset
# 
# The DICOM format is so cool, but I prefer normal images :)
# 
# With 156GB (compressed) it is very difficult to work with the resources of the vast majority of the mortals.
# This notebook shows you how to scale down all the images and create a new dataset easier to deal with.
# Even with the best computing resources, I don't think it's necessary to use the original size to get good accuracy.
# 
# IMPORTANT: In this notebook runs in a subset of the data, so don't use the generated output. That is because the Kaggle notebook runs out of space if you use all the examples. If you want to run this by yourself you should run it in a different machine or opening the next notebook https://colab.research.google.com/gist/guiferviz/50912a681776d5afe012b1a9259bd637/resize-dataset.ipynb in Google Colab. If you try to unzip the data in Google Colab you will also run out of space, so I've used the amazing tool *fuse-zip* to mount the zip and work with the files in it without extracting any of those.
# 
# Some code taken from:
# * https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
# * https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109649#latest-631701
# 
# # Constants

# In[ ]:


# Desired output size.
RESIZED_WIDTH, RESIZED_HEIGHT = 128, 128

OUTPUT_FORMAT = "png"

OUTPUT_DIR = "output"


# # Imports

# In[ ]:


import glob

import joblib

import numpy as np

import PIL

import pydicom

import tqdm


# # Get images paths

# In[ ]:


data_dir = "../input/rsna-intracranial-hemorrhage-detection"
get_ipython().system('ls {data_dir}')


# In[ ]:


train_dir = "stage_1_train_images"
train_paths = glob.glob(f"{data_dir}/{train_dir}/*.dcm")
test_dir = "stage_1_test_images"
test_paths = glob.glob(f"{data_dir}/{test_dir}/*.dcm")
len(train_paths), len(test_paths)


# # Preprocess all data
# 
# First declare a bunch of useful functions.

# In[ ]:


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "window_center": img_dicom.WindowCenter,
        "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img 

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(f"{OUTPUT_DIR}/{subfolder}/{name}.{OUTPUT_FORMAT}")

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def prepare_image(img_path):
    img_dicom = pydicom.read_file(img_path)
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255
    img_pil = resize(img, RESIZED_WIDTH, RESIZED_HEIGHT)
    return img_id, img_pil

def prepare_and_save(img_path, subfolder):
    try:
        l.error("loading eso")
        img_id, img_pil = prepare_image(img_path)
        save_img(img_pil, subfolder, img_id)
    except KeyboardInterrupt:
        # Rais interrupt exception so we can stop the cell execution
        # without shutting down the kernel.
        raise
    except:
        l.error(f"Error processing the image: {img_path}")

def prepare_images(imgs_path, subfolder):
    for i in tqdm.tqdm(imgs_path):
        prepare_and_save(i, subfolder)
import logging as l
def prepare_images_njobs(img_paths, subfolder, n_jobs=-1):
    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(i, subfolder) for i in tqdm.tqdm(img_paths))


# In[ ]:


get_ipython().system('mkdir -p {OUTPUT_DIR}/{train_dir}')
get_ipython().system('mkdir -p {OUTPUT_DIR}/{test_dir}')


# In[ ]:


# Running on the first 100 files of train and set!!!
prepare_images_njobs(train_paths[:100], train_dir)
prepare_images_njobs(test_paths[:100], test_dir)


# # Load converted images
# 
# Let's test that everything is ok!

# In[ ]:


train_output_path = glob.glob(f"{OUTPUT_DIR}/{train_dir}/*")


# In[ ]:


img_path = train_output_path[0]
PIL.Image.open(img_path)


# # Comments and future work
# 
# Keep in mind that you may want to normalize the downsampled images before feeding them into a neural network, the values are between 0 and 255.
# 
# Finally, a series of ideas/future work/open questions:
# * Maybe we can let the algorithm to optimize the window size and width.
# * Create differents datasets with different sizes and try to find the best training time and memory footprint vs accuracy.
# * Try to crop the images. Black margins are too big. Not sure if this change of scale can affect the algorithm. If it does not affect it will allow to use smaller images with the same details.
# * Can we use '(0020, 0032) Image Position (Patient)' and '(0020, 0037) Image Orientation (Patient)' to rotate and crop the images? For example, the image with "ID_c03cdcb55" has a big rotation.
