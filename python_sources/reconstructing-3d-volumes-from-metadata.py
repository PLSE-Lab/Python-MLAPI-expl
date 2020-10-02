#!/usr/bin/env python
# coding: utf-8

# # Reconstructing 3D volumes from metadata
# In this kernel we will use the `StudyInstanceUID` in the metadata to group together images from the same scan. We will then sort the images using `ImagePositionPatient_2` and create 3D volumes. The 2D DICOM images supplied to us are *axial* slices. Once we have a 3D volume we will be able to make *sagittal* and *coronal* slices
# 
# ![](https://www.radiologycafe.com/images/basics/ct-planes.png)
# 
# Credit due to this discussion https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109953#latest-639253

# In[ ]:


import os
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pylab as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')

data_path = "../input/rsna-intracranial-hemorrhage-detection"
metadata_path = "../input/rsna-ich-metadata"
os.listdir(metadata_path)


# # Prepare the labels & metadata
# The metadata was extracted beforehand using `pydicom`. This takes a while so I saved the results in these parquet files so they don't need to be generated each time.

# In[ ]:


train_df = pd.read_csv(f'{data_path}/stage_1_train.csv').drop_duplicates()
train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")
train_labels.head()


# In[ ]:


def get_metadata(image_dir):

    labels = [
        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID', 
        'WindowCenter', 'WindowWidth', 'Image',
    ]

    data = {l: [] for l in labels}

    for image in tqdm_notebook(os.listdir(image_dir)):
        data["Image"].append(image[:-4])

        ds = pydicom.dcmread(os.path.join(image_dir, image))

        for metadata in ds.dir():
            if metadata != "PixelData":
                metadata_values = getattr(ds, metadata)
                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:
                    for i, v in enumerate(metadata_values):
                        data[f"{metadata}_{i}"].append(v)
                else:
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:
                        data[metadata].append(metadata_values[0])
                    else:
                        data[metadata].append(metadata_values)

    return pd.DataFrame(data).set_index("Image")


# In[ ]:


# Generate metadata dataframes
train_metadata = get_metadata(os.path.join(data_path, "stage_1_train_images"))
test_metadata = get_metadata(os.path.join(data_path, "stage_1_test_images"))

train_metadata.to_parquet(f'{data_path}/train_metadata.parquet.gzip', compression='gzip')
test_metadata.to_parquet(f'{data_path}/test_metadata.parquet.gzip', compression='gzip')


# In[ ]:


train_metadata = pd.read_parquet(f'{metadata_path}/train_metadata.parquet.gzip')
test_metadata = pd.read_parquet(f'{metadata_path}/test_metadata.parquet.gzip')

train_metadata["Dataset"] = "train"
test_metadata["Dataset"] = "test"

train_metadata = train_metadata.join(train_labels)

metadata = pd.concat([train_metadata, test_metadata], sort=True)
metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False)
metadata.head()


# In[ ]:


metadata["StudyInstanceUID"].nunique()


# # Group the unique studies

# In[ ]:


studies = metadata.groupby("StudyInstanceUID")
studies_list = list(studies)

study_name, study_df = studies_list[0]
study_df.head()


# In[ ]:


studies.size().describe()


# In[ ]:


plt.hist(studies.size());


# It seems like the unique studies can have anywhere between 20 and 60 images (i.e. axial slices). Perhaps they need to be resized to a constant z-dimension? Or if x & y is constant, use an architecture that can cope with this e.g. adaptive pooling layers?
# 
# Let's check if any studies straddle train/test:

# In[ ]:


studies.filter(lambda x: x["Dataset"].nunique() > 1)


# Looks like none - which is good, however we still need to remember that patients can have multiple scans and these can be across the train & stage 1 test.

# # Create a 3D volume for a single study
# We'll use the first study in the grouped `studies` which is comprised of 40 individial axial DICOM images
# 
# Thanks to this notebook for the windowing functions https://www.kaggle.com/wfwiggins203/eda-dicom-tags-windowing-head-cts

# In[ ]:


def window_img(dcm, width=None, level=None, norm=True):
    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Pad non-square images
    if pixels.shape[0] != pixels.shape[1]:
        (a,b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a-b) // 2, (a-b) // 2))
        else:
            padding = (((b-a) // 2, (b-a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)
            
    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)
    
    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img


# In[ ]:


volume, labels = [], []
for index, row in study_df.iterrows():
    if row["Dataset"] == "train":
        dcm = pydicom.dcmread(os.path.join(data_path, "stage_1_train_images", index+".dcm"))
    else:
        dcm = pydicom.dcmread(os.path.join(data_path, "stage_1_test_images", index+".dcm"))
        
    img = window_img(dcm)
    label = row[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]]
    volume.append(img)
    labels.append(label)
    
volume = np.array(volume)
labels = np.array(labels)


# In[ ]:


volume.shape, labels.shape


# The provided DICOM images are axial slices. Let's use our new 3D volume to create sagittal and coronal slices
# * Red line - axial plane
# * Green line - sagittal plane
# * Blue line - coronal plane

# In[ ]:


# Axial
plt.figure(figsize=(8, 8))
plt.imshow(volume[20, :, :], cmap=plt.cm.bone)
plt.vlines(300, 0, 512, colors='g')
plt.hlines(300, 0, 512, colors='b');


# In[ ]:


# Sagittal
plt.figure(figsize=(8, 8))
plt.imshow(volume[:, :, 300], aspect=9, cmap=plt.cm.bone)
plt.vlines(300, 0, 40, colors='b')
plt.hlines(20, 0, 512, colors='r');


# In[ ]:


# Coronal
plt.figure(figsize=(8, 8))
plt.imshow(volume[:, 300, :], aspect=9, cmap=plt.cm.bone)
plt.vlines(300, 0, 40, colors='g')
plt.hlines(20, 0, 512, colors='r');


# 
