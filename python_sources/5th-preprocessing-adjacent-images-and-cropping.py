#!/usr/bin/env python
# coding: utf-8

# # Preprocessing: Spatially adjacent RGB images and cropping
# This is a follow on from my previous notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
# 
# This is also a part of our 5th place solution https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117232#latest-672657

# In[ ]:


import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image, ImageFile
import matplotlib.pylab as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')

data_path = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
metadata_path = "../input/rsna-ich-metadata"
os.listdir(metadata_path)


# In[ ]:


os.listdir(data_path)


# Load the metadata (see my previous notebook to see how this dataset was generated)
# 
# Dataset: https://www.kaggle.com/anjum48/rsna-ich-metadata

# In[ ]:


train_metadata = pd.read_parquet(f'{metadata_path}/train_metadata.parquet.gzip')
test_metadata = pd.read_parquet(f'{metadata_path}/test_metadata.parquet.gzip')


# # Create triplets of images
# Based on `StudyInstanceUID` and sorting on `ImagePositionPatient` it is possible to reconstruct 3D volumes for each study. However since each study contained a variable number of axial slices (between 20-60) this makes it difficult to create a architecture that implements 3D convolutions. Instead, triplets of images were created from the 3D volumes to represent the RGB channels of an image, i.e. the green channel being the target image and the red & blue channels being the adjacent images. If an image was at the edge of the volume, then the green channel was repeated. This is essentially a 3D volume but only using 3 axial slices. At this stage no windowing was applied and the image is retained in Hounsfield units.

# In[ ]:


def build_triplets(metadata):
    metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False)
    studies = metadata.groupby("StudyInstanceUID")
    triplets = []

    for study_name, study_df in tqdm_notebook(studies):
        padded_names = np.pad(study_df.index, (1, 1), 'edge')

        for i, img in enumerate(padded_names[1:-1]):
            t = [padded_names[i], img, padded_names[i + 2]]
            triplets.append(t)

    return pd.DataFrame(triplets, columns=["red", "green", "blue"])


# In[ ]:


train_triplets = build_triplets(train_metadata)
test_triplets = build_triplets(test_metadata)
train_triplets.to_csv("train_triplets.csv")
test_triplets.to_csv("stage_1_test_triplets.csv")


# In[ ]:


train_triplets.head()


# # Creating the RGB image
# Let's construct an RGB image using these triplets

# In[ ]:


def prepare_dicom(dcm):
    """
    Converts a DICOM object to a 16-bit Numpy array (in Hounsfield units)
    :param dcm: DICOM Object
    :return: Numpy array in int16
    """

    try:
        # https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        if dcm.BitsStored == 12 and dcm.PixelRepresentation == 0 and dcm.RescaleIntercept > -100:
            x = dcm.pixel_array + 1000
            px_mode = 4096
            x[x >= px_mode] = x[x >= px_mode] - px_mode
            dcm.PixelData = x.tobytes()
            dcm.RescaleIntercept = -1000

        pixels = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
    except ValueError as e:
        print("ValueError with", dcm.SOPInstanceUID, e)
        return np.zeros((512, 512))

    # Pad the image if it isn't square
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

    return pixels.astype(np.int16)


# In[ ]:


channels = train_triplets.iloc[3079]
channels


# In[ ]:


rgb = []

for image_id in channels:
    dcm = pydicom.dcmread(os.path.join(data_path, "stage_2_train", image_id + ".dcm"))
    rgb.append(prepare_dicom(dcm))
    
img = np.stack(rgb, -1)
img = np.clip(img, 0, 255).astype(np.uint8) 


# On the final line above, we actually clip the image between 0 & 255 Hounfield units. This is ok, since most of the important features in the image are between this range.
# 
# Since the image is still technically in Hounsfield units, you can apply a window to it later (e.g. brain, subdural etc.), however since we have already clipped this image, this may impact the bone window.
# 
# Let's check out our image:

# In[ ]:


plt.figure(figsize=(8, 8))
plt.imshow(img);


# # Cropping the image
# As you can see there is a) a lot of black space which we don't want to waste our valuable GPU operations on and b) the CT scanner headrest. Let's get rid of both of these using `scipy.ndimage.label`

# In[ ]:


from scipy import ndimage


# In[ ]:


labeled_blobs, number_of_blobs = ndimage.label(img)
blob_sizes = np.bincount(labeled_blobs.flatten())
number_of_blobs


# In[ ]:


blob_sizes


# This labels adjacent "blobs" in the image, i.e. groups of connected pixels.
# 
# In this example we have 39 blobs, each with different sizes. Lets see the first blob:

# In[ ]:


blob_0 = labeled_blobs == 0  # label 0 has 593185 connected pixels
blob_0 = np.max(blob_0, axis=-1)

plt.figure(figsize=(8, 8))
plt.imshow(blob_0);


# Ok, the largest blob appears to be the background. Let's try the next largest blob

# In[ ]:


blob_1 = labeled_blobs == 1  # label 1 has 173243 connected pixels
blob_1 = np.max(blob_1, axis=-1)

plt.figure(figsize=(8, 8))
plt.imshow(blob_1);


# Great! We've found that the **2nd largest blob** is the head and there is no headrest. We can use this mask to define our cropping extent (`x_min, x_max, y_min, y_max`) and create a cropping function

# In[ ]:


class CropHead(object):
    def __init__(self, offset=10):
        """
        Crops the head by labelling the objects in an image and keeping the second largest object (the largest object
        is the background). This method removes most of the headrest

        Originally made as a image transform for use with PyTorch, but too slow to run on the fly :(
        :param offset: Pixel offset to apply to the crop so that it isn't too tight
        """
        self.offset = offset

    def crop_extents(self, img):
        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            labeled_blobs, number_of_blobs = ndimage.label(img_array)
            blob_sizes = np.bincount(labeled_blobs.flatten())
            head_blob = labeled_blobs == np.argmax(blob_sizes[1:]) + 1  # The number of the head blob
            head_blob = np.max(head_blob, axis=-1)

            mask = head_blob == 0
            rows = np.flatnonzero((~mask).sum(axis=1))
            cols = np.flatnonzero((~mask).sum(axis=0))

            x_min = max([rows.min() - self.offset, 0])
            x_max = min([rows.max() + self.offset + 1, img_array.shape[0]])
            y_min = max([cols.min() - self.offset, 0])
            y_max = min([cols.max() + self.offset + 1, img_array.shape[1]])

            return x_min, x_max, y_min, y_max
        except ValueError:
            return 0, 0, -1, -1

    def __call__(self, img):
        """
        Crops a CT image to so that as much black area is removed as possible
        :param img: PIL image
        :return: Cropped image
        """

        x_min, x_max, y_min, y_max = self.crop_extents(img)

        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            return Image.fromarray(np.uint8(img_array[x_min:x_max, y_min:y_max]))
        except ValueError:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(offset={})'.format(self.offset)


# In[ ]:


crop_head = CropHead()
img_cropped = crop_head(img)


# # The final product
# Let's check our cropped image

# In[ ]:


plt.figure(figsize=(8, 8))
plt.imshow(img_cropped);


# Looks good! Most of the black space is removed and most of the headrest is gone. The image can then be resized into a square (e.g. 224x224) for use in a CNN

# In[ ]:




