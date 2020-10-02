#!/usr/bin/env python
# coding: utf-8

# This notebook takes a look at the data that will be used to train a model to predict Melanoma Skin Lessions. It focuses on work to understand/learn the DICOM data format as it it is used in the Kaggle ISIC 2020 competition.  It also shows work to visualize the embedded images in a more *natural* (RGB) format vs. the embedded `YBR_FULL_422` format that if displayed as is produces images that in the author's opinion do not look natural.
# 
# The Kaggle contest (https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion?sortBy=relevance&group=all&search=color&page=1&pageSize=20&category=all)has a very large image dataset. This notebook uses a small subset of the original dataset (a few images from the `test` directory)
# 
# This notebook does not include any modeling to predict melanoma. It only covers the work to open DICOM files, looking at the metadata, manipulating it if needed and displaying the data for visual analysis.

# # References

# * `pydicom` https://pydicom.github.io/pydicom/stable/old/working_with_pixel_data.html#dataset-pixel-array
# * fast.ai V2. https://dev.fast.ai/
# * Fastai2 DICOM Starer. https://www.kaggle.com/avirdee/fastai2-dicom-starter
# * Choosing Colormaps in Matplotlib (`plt`). https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# * YUV to RGB conversion. http://www.fourcc.org/fccyvrgb.php
# * DICOM is easy!. https://dicomiseasy.blogspot.com/2012/08/chapter-12-pixel-data.html
# * Convert PIL image to OpenCV Image. https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
# * Convert OpenCV Image to PIL. https://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library

# # Background

# DICOM Images (Digital Imaging and Communication in medicines) are the standard that is used in medicine to allow images from X-Ray, MRI, Cat Scan and others (along with metadadat) to be exchanged among medical professionals.
# 
# In python, DICOM images can be viewed and analyzed using the  `pydicom` package. If you have installed `fastai V2` then this package will have already be installed.
# 
# There is also a need to use a `PyTorch` computer vision package called `Kornia` however if using `fastai v2` this pakcage is also part of the distribution.
# 
# There are other packages that will be needed. They are installed in the`conda`environment used used in this project. To see what packages are part of this environment please see the `requirements.yml` file in the root directory of this project.

# # Initialize

# Load packages we need

# In[ ]:


get_ipython().system('pip install --upgrade fastai2 > /dev/null')


# In[ ]:


from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *


# In[ ]:


import pydicom


# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


import cv2


# Get the files

# In[ ]:


source = Path("../input/siim-isic-melanoma-classification")
files = os.listdir(source)
files


# In[ ]:


train = source/'train'
train_files = get_dicom_files(train)
train_files


# Pick one of the files and see what information is in the file

# In[ ]:


image = train_files[42]


# In[ ]:


dimg = dcmread(image)


# Info for the image

# In[ ]:


dimg


# Important fields:
# 
# * **Pixel Data** is where the actual image is stored. The order of pixels encoded for each image plane is left to right, top to bottom, i.e., the upper left pixel (labeled 1,1) is encoded first
# 
# * **Photometric Interpretation** This is the color space. From what I see there, these images are not in the usual RGB format, but they are in `YBR_FULL_422`. As I understand things today if I use `fast.ai v2` medical images to display the images, the will not look like what you expect a picture of a skin lession to look like.  I don't believe it matters when the model is trained, but later in this notebook after I extract th image I use `Open CV` to convert the color space to `RGB`
# 
# * **Samples per Pixel** One for monochrome and 3 for RGB (or in this case for `YBR_FULL_422`) images

# Pydicom reads pixel data as the raw bytes found in the file and most of the time the `Pyxel Data` is not useable as read since the data can be stored in one of several formats:
# 
# * Unsigned integers, or floats
# * There may be multiple image frames
# * There may be multiple planes per frame (i.e. RGB) and the order of the pixels may be different.
# 
# see the `pydicom` reference page for other possible formats.

# This is a function (I found this in this notebook in Kaggle: https://www.kaggle.com/avirdee/fastai2-dicom-starter) that will display an image and show choosen tags within the head of the DICOM, in this case PatientName, PatientID, PatientSex, BodyPartExamined and we can use Tranform from fastai2 that conveniently allows us to resize the image. Note that this function (as I understand things now) does not handle the display of the `YBR_FULL_422` images and when they are displayed, they will not look like you would expect a skim image to look like.  I will later in the notebook create a new function that will show the images with the *usual* colors (`RGB` color space)

# In[ ]:


def show_one_patient(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'patient Name: {pat.PatientName}')
    print(f'Patient ID: {pat.PatientID}')
    print(f'Patient age: {pat.PatientAge}')
    print(f'Patient Sex: {pat.PatientSex}')
    print(f'Body part: {pat.BodyPartExamined}')
    trans = Transform(Resize(256))
    dicom_create = PILDicom.create(file)
    dicom_transform = trans(dicom_create)
    return show_image(dicom_transform)


# Now let,s Use `fast.ai v2` function to open and read a `DICOM1 file

# In[ ]:


patient = dcmread(image)


# Look at the color space

# In[ ]:


print(f'Photometric Interpretation: {patient.PhotometricInterpretation}')


# Get the image from the DICOM file as a PIL file

# In[ ]:


dicom_create = PILDicom.create(image)


# Display the image. Since PIL (as I understand) only handles RGB images (or maybe there is another option that needs to be used that I know know about), the image does not look as one would excpet a skin picture

# In[ ]:


dicom_create.show(figsize=(6,6), cmap=plt.cm.gist_ncar)


# I will use `OpenCV` to convert the `YBR_FULL_422` image to RGB. To do this I will extract from the `DICOM` file the image, which as we saw earlier is stored in `pixel_array`. We take the pixel data array and convert it to a `PIL` image (there is probably a better way to do the following stes, but I used this approach to to experiment, so I kept these steps)

# In[ ]:


pil_image = Image.fromarray(dcmread(image).pixel_array)


# Now convert the PIL image to an OpenCV image. Note that the OpenCV image will be in BGR.

# In[ ]:


open_cv_image = np.array(pil_image) 


# According to https://dicomiseasy.blogspot.com/2012/08/chapter-12-pixel-data.html, `YBR_FULL_422` means that the color channels are in the YCbCr color space that is used in JPEG. The next line uses the OpenCV `cv2.cvtColor` to transform the color space.

# In[ ]:


open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)


# Now that we have an OpenCV image in the right space, we convert it back to PIL to Display (note that I could have used OpenCV to display it, but since `fast.ai` uses PIL, I wanted to stay using `PIL`)

# In[ ]:


pil_image = Image.fromarray(open_cv_image)


# Now when we display the image, it will look like you would expect a skin picture to look like.

# In[ ]:


plt.imshow(pil_image)


# Now that I know how to display the image embedded in the DICOM file in a natural format, I can create a new function to use in other notebooks rived from the show_one_patient in https://www.kaggle.com/avirdee/fastai2-dicom-starter

# In[ ]:


def show_one_patient_RGB(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'patient Name: {pat.PatientName}')
    print(f'Patient ID: {pat.PatientID}')
    print(f'Patient age: {pat.PatientAge}')
    print(f'Patient Sex: {pat.PatientSex}')
    print(f'Body part: {pat.BodyPartExamined}')
    trans = Transform(Resize(256))
    
    pil_image = Image.fromarray(dcmread(image).pixel_array)
    # Not sure yet about the use of the following line. For now uncommented.
    # pil_image = trans(pil_image)
    open_cv_image = np.array(pil_image) 
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)
    pil_image = Image.fromarray(open_cv_image)
    # Could add a parameter to specify figize. For now it is set to (6,6)
    
    return show_image(pil_image, figsize=(6,6))


# In[ ]:


show_one_patient_RGB(image)


# As we can see, I have now the skin image in more *natural* colors or as I would expect to seem them from a photograph from the doctor's office.

# In[ ]:




