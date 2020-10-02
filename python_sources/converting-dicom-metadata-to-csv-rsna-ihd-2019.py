#!/usr/bin/env python
# coding: utf-8

# ## Converting DICOM metadata to CSV files

# ![](https://www.rsna.org/-/media/Images/RSNA/Menu/logo_sml.ashx?w=100&la=en&hash=9619A8238B66C7BA9692C1FC3A5C9E97C24A06E1)

# As many of you may have noticed, the DICOM files for the [RSNA Intracranial Hemorrhage competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) contain metadata about the image data. This metadata may or may not be useful for the competition. In this kernel you'll find a convenient overview of the metadata and a script which adds the metadata to stage_1 csv files. The CSV files which includes the metadata can be found in "output files" below.
# 
# If you are interested in getting the preprocessed competition CSV files that are ready for training I suggest you check out [this Kaggle kernel](https://www.kaggle.com/carlolepelaars/preprocessing-csv-s-for-training-rsna-ih-2019).

# ## Table Of Contents

# - [Dependencies](#1)
# - [Preparation](#2)
# - [Metadata](#3)
# - [Type Conversion](#4)
# - [Merge and Save](#5)
# - [Final Check](#6)

# ## Dependencies <a id="1"></a>

# In[ ]:


# Standard libraries
import os
import gc
import pydicom # For accessing DICOM files
import numpy as np
import pandas as pd 
import random as rn
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths 
KAGGLE_DIR = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
IMG_PATH_TRAIN = KAGGLE_DIR + 'stage_2_train/'
IMG_PATH_TEST = KAGGLE_DIR + 'stage_2_test/'
TRAIN_CSV_PATH = KAGGLE_DIR + 'stage_2_train.csv'
TEST_CSV_PATH = KAGGLE_DIR + 'stage_2_sample_submission.csv'

# Seed for reproducability
seed = 1234
np.random.seed(seed)
rn.seed(seed)


# In[ ]:


# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(KAGGLE_DIR)[2:]:
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))


# ## Preparation <a id="2"></a>

# In[ ]:


# Load in raw datasets
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
# For convenience, collect sub type and seperate PatientID as new features
for df in [train_df, test_df]:
    df['Sub_type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
    df['PatientID'] = df['ID'].str.split("_", n = 3, expand = True)[1]


# In[ ]:


# All filenames for train and test images
train_images = os.listdir(IMG_PATH_TRAIN)
test_images = os.listdir(IMG_PATH_TEST)


# ## Metadata <a id="3"></a>

# The [pydicom](https://pydicom.github.io/pydicom/stable/getting_started.html) library allows us to conveniently read in DICOM files and access different values from the file. The actual image can be found in "pixel_array".

# In[ ]:


print('Example of all data in a single DICOM file:\n')
example_dicom = pydicom.dcmread(IMG_PATH_TRAIN + train_images[0])
print(example_dicom)


# In[ ]:


# All columns for which we want to collect information
meta_cols = ['BitsAllocated','BitsStored','Columns','HighBit',
             'Modality','PatientID','PhotometricInterpretation',
             'PixelRepresentation','RescaleIntercept','RescaleSlope',
             'Rows','SOPInstanceUID','SamplesPerPixel','SeriesInstanceUID',
             'StudyID','StudyInstanceUID','ImagePositionPatient',
             'ImageOrientationPatient','PixelSpacing']


# In[ ]:


# Initialize dictionaries to collect the metadata
col_dict_train = {col: [] for col in meta_cols}
col_dict_test = {col: [] for col in meta_cols}


# Here we extract all features for the training and testing set.

# In[ ]:


# Get values for training images
for img in tqdm(train_images): 
    dicom_object = pydicom.dcmread(IMG_PATH_TRAIN + img)
    for col in meta_cols: 
        col_dict_train[col].append(str(getattr(dicom_object, col)))

# Store all information in a DataFrame
meta_df_train = pd.DataFrame(col_dict_train)
del col_dict_train
gc.collect()


# In[ ]:


# Get values for test images
for img in tqdm(test_images): 
    dicom_object = pydicom.dcmread(IMG_PATH_TEST + img)
    for col in meta_cols: 
        col_dict_test[col].append(str(getattr(dicom_object, col)))

# Store all information in a DataFrame
meta_df_test = pd.DataFrame(col_dict_test)
del col_dict_test
gc.collect()


# ## Type Conversion <a id="4"></a>

# Above we used a bit of a hacky solution by converting all metadata to string values. Now we will convert all features back to proper types.
# 
# All numeric features will be converted to float types. We will keep all categorical features as string types.
# 
# The 'WindowCenter' and 'WindowWidth' were rather odd as they featured both int, float and list values. For now I skipped these features, but I may add them to this kernel later. Feel free to share code to conveniently handle this data.
# 
# The features 'ImagePositionPatient', 'ImageOrientationPatient' and 'PixelSpacing' are stored as lists. In order to easily access these features we create a new column for every value in the list. 
# 
# We fill missing values with values that are outside the range of the feature (-999).
# 
# N.B.: I'm using a quite hacky solution to parse all the values into a DataFrame. Feel free to suggest more elegant ways to get the same result if you want to help in improving this kernel.

# In[ ]:


# Specify numeric columns
num_cols = ['BitsAllocated', 'BitsStored','Columns','HighBit', 'Rows',
            'PixelRepresentation', 'RescaleIntercept', 'RescaleSlope', 'SamplesPerPixel']


# In[ ]:


# Split to get proper PatientIDs
meta_df_train['PatientID'] = meta_df_train['PatientID'].str.split("_", n = 3, expand = True)[1]
meta_df_test['PatientID'] = meta_df_test['PatientID'].str.split("_", n = 3, expand = True)[1]

# Convert all numeric cols to floats
for col in num_cols:
    meta_df_train[col] = meta_df_train[col].fillna(-9999).astype(float)
    meta_df_test[col] = meta_df_test[col].fillna(-9999).astype(float)


# In[ ]:


# Hacky solution for multi features
for df in [meta_df_train, meta_df_test]:
    # ImagePositionPatient
    ipp1 = []
    ipp2 = []
    ipp3 = []
    for value in df['ImagePositionPatient'].fillna('[-9999,-9999,-9999]').values:
        value_list = eval(value)
        ipp1.append(float(value_list[0]))
        ipp2.append(float(value_list[1]))
        ipp3.append(float(value_list[2]))
    df['ImagePositionPatient_1'] = ipp1
    df['ImagePositionPatient_2'] = ipp2
    df['ImagePositionPatient_3'] = ipp3
    
    # ImageOrientationPatient
    iop1 = []
    iop2 = []
    iop3 = []
    iop4 = []
    iop5 = []
    iop6 = []
    # Fill missing values and collect all Image Orientation information
    for value in df['ImageOrientationPatient'].fillna('[-9999,-9999,-9999,-9999,-9999,-9999]').values:
        value_list = eval(value)
        iop1.append(float(value_list[0]))
        iop2.append(float(value_list[1]))
        iop3.append(float(value_list[2]))
        iop4.append(float(value_list[3]))
        iop5.append(float(value_list[4]))
        iop6.append(float(value_list[5]))
    df['ImageOrientationPatient_1'] = iop1
    df['ImageOrientationPatient_2'] = iop2
    df['ImageOrientationPatient_3'] = iop3
    df['ImageOrientationPatient_4'] = iop4
    df['ImageOrientationPatient_5'] = iop5
    df['ImageOrientationPatient_6'] = iop6
    
    # Pixel Spacing
    ps1 = []
    ps2 = []
    # Fill missing values and collect all pixal spacing features
    for value in df['PixelSpacing'].fillna('[-9999,-9999]').values:
        value_list = eval(value)
        ps1.append(float(value_list[0]))
        ps2.append(float(value_list[1]))
    df['PixelSpacing_1'] = ps1
    df['PixelSpacing_2'] = ps2


# ## Merge and Save <a id="5"></a>

# This metadata will only be useful if we can connect it to specific images. To make sure every value is in the correct row we can conveniently merge on the PatientID feature. However, an inner or left join will not work since our DataFrame with metadata contains a lot of rows that are not in the original DataFrame. Joining on the right and using a few columns from the original DataFrame will do the trick.

# In[ ]:


# Merge DataFrames
train_df_merged = meta_df_train.merge(train_df, how='left', on='PatientID')
train_df_merged['ID'] = train_df['ID']
train_df_merged['Label'] = train_df['Label']
train_df_merged['Sub_type'] = train_df['Sub_type']
test_df_merged = meta_df_test.merge(test_df, how='left', on='PatientID')
test_df_merged['ID'] = test_df['ID']
test_df_merged['Label'] = test_df['Label']
test_df_merged['Sub_type'] = test_df['Sub_type']


# In[ ]:


# Save to CSV
train_df_merged.to_csv('stage_2_train_with_metadata.csv', index=False)
test_df_merged.to_csv('stage_2_test_with_metadata.csv', index=False)


# ## Final Check <a id="6"></a>

# In[ ]:


# Final check on the new dataset
print('Training Data:')
display(train_df_merged.head(3))
display(train_df_merged.tail(3))
print('Testing Data:')
display(test_df_merged.head(3))
display(test_df_merged.tail(3))


# That's all! You can find the new CSV's with the metadata in the "output files" of this kernel.
# 
# If you like this Kaggle kernel, feel free to give an upvote and leave a comment! I will try to implement your suggestions!
