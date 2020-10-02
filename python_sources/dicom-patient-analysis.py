#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pydicom

from tqdm import tqdm_notebook


# In[ ]:


# Get directory names/locations
data_root = os.path.abspath("../input/rsna-intracranial-hemorrhage-detection/")

train_img_root = data_root + "/stage_1_train_images/"
test_img_root  = data_root + "/stage_1_test_images/"

train_labels_path = data_root + "/stage_1_train.csv"
test_labels_path  = data_root + "/stage_1_test.csv"

# Create list of paths to actual training data
train_img_paths = os.listdir(train_img_root)
test_img_paths  = os.listdir(test_img_root)

# Dataset size
num_train = len(train_img_paths)
num_test  = len(test_img_paths)


# In[ ]:


def create_efficient_df(data_path):
    
    # Define the datatypes we're going to use
    final_types = {
        "ID": "str",
        "Label": "float16"
    }
    features = list(final_types.keys())
    
    # Use chunks to import the data so that less efficient machines can only use a 
    # specific amount of chunks on import
    df_list = []

    chunksize = 1_000_000

    for df_chunk in pd.read_csv(data_path, dtype=final_types, chunksize=chunksize): 
        df_list.append(df_chunk)
        
    df = pd.concat(df_list)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    del df_list

    return df

train_labels_df = create_efficient_df(train_labels_path)
train_labels_df[train_labels_df["Label"] > 0].head()


# In[ ]:


hem_types = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any"
]

new_cols = [
    "id",
    "type_0",
    "type_1",
    "type_2",
    "type_3",
    "type_4",
    "type_5"
]

num_ids = int(train_labels_df.shape[0] / len(hem_types))
print("Number of unique patient IDs: {}".format(num_ids))

empty_array = np.ones((num_ids, len(new_cols)))
hem_df = pd.DataFrame(data=empty_array, columns=new_cols)

# Fill in the ID of each image
hem_df["id"] = list(train_labels_df.iloc[::len(hem_types)]["ID"].str.split(pat="_").str[1])
    
# Fill in the categorical columns of each image
for hem_ix, hem_col in enumerate(list(hem_df)[1:]):
    hem_df[hem_col] = list(train_labels_df.iloc[hem_ix::len(hem_types), 1])
    
hem_df.info()
hem_df[hem_df["type_5"] > 0].head()


# # Introduction
# 
# Hello there! My name is Mick and I'm continuing my analysis of the RSNA Intracranial Hemorrhage dataset. I'll be analyzing the dataset of DICOM images in conjunction with Pydicom, a Python package specifically for parsing .dcm files. This notebook is going to be all about the DICOM file format and trying to parse out clinical, diagnostic, and locational information from DICOM files.
# 
# If you've read [my EDA notebook for this dataset](https://www.kaggle.com/smit2300/hemorrhage-eda-encoding-dicom-introduction) you can see how I encoded the dataset labels to be a bit more analysis friendly. If you dont want to look through that then the methods I used are recreated above in this notebook.
# 
# So with that bit of preprocessing knowledge out of the way let's get into some DICOM analysis!

# In[ ]:


hem_df.head()


# # DICOM Refresher
# 
# This is explained in a bunch of other notebooks throughout this competition, but briefly: a DICOM image is one created specifically for sending medical images between clinicians that contain contextual information for the clinician. The purpose of the format is to send an image that contains some of its own labeling information directly in the file. Below is an example of the info in a DICOM image for this dataset.

# In[ ]:


random_ix = random.randint(0, len(train_img_paths))
random_path = train_img_root + train_img_paths[random_ix]

dcm_info = pydicom.dcmread(random_path)
print("===IMAGE MEDICAL INFO===")
print(dcm_info)


# # DICOM Information DataFrame
# 
# Now that we know how to extract information from a .dcm file we can choose which of these data fields is going to be most useful for our prediction of intracranial hemorrhage. We can enrich our exploration of this dataset by splitting out the dataset by patient, rather than just individual image ID as well. If we get enough images of a patient then we can try to recreate a clinical trial where a patient is being imaged for this study. Then based on the positional information in the .dcm files we can possible do some scan recreation. Let's expand our `hem_df` dataframe to include the following columns in addition to our hemorrhage labels:
#  * Image ID
#  * Patient ID
#  * Position of scan image in patient
#  * Orientation of scan image in patient

# In[ ]:


DEV_RUN = True

if DEV_RUN:
    SET_SIZE = 50_000
    print("Creating {} element subset of hemorrhage dataset".format(SET_SIZE))
    hem_df = hem_df.iloc[:SET_SIZE, :]
    
patient_ids  = np.zeros((hem_df.shape[0],))
positions    = np.zeros((hem_df.shape[0]))
orientations = np.zeros((hem_df.shape[0]))

hem_df["patient_id"]    = patient_ids
hem_df["position_0"]    = positions
hem_df["position_1"]    = positions
hem_df["position_2"]    = positions
hem_df["orientation_0"] = orientations
hem_df["orientation_1"] = orientations
hem_df["orientation_2"] = orientations
hem_df["orientation_3"] = orientations
hem_df["orientation_4"] = orientations
hem_df["orientation_5"] = orientations

del patient_ids
del positions
del orientations

for row_ix, row in tqdm_notebook(hem_df.iterrows()):
    
    full_path = train_img_root + "ID_" + row["id"] + ".dcm"
    dcm_info  = pydicom.dcmread(full_path)
    
    patient_id  = dcm_info.PatientID.split("_")[1]
    position    = dcm_info.ImagePositionPatient
    orientation = dcm_info.ImageOrientationPatient
        
    hem_df["patient_id"].iloc[row_ix]  = patient_id
    
    hem_df["position_0"].iloc[row_ix]    = position[0]
    hem_df["position_1"].iloc[row_ix]    = position[1]
    hem_df["position_2"].iloc[row_ix]    = position[2]
    
    hem_df["orientation_0"].iloc[row_ix] = orientation[0]
    hem_df["orientation_1"].iloc[row_ix] = orientation[1]
    hem_df["orientation_2"].iloc[row_ix] = orientation[2]
    hem_df["orientation_3"].iloc[row_ix] = orientation[3]
    hem_df["orientation_4"].iloc[row_ix] = orientation[4]
    hem_df["orientation_5"].iloc[row_ix] = orientation[5]
        
hem_df.head()


# ## Analyzing Duplicates
# 
# Now that we have more metadata loaded into our dataframe, we can start to analyze based on each patient. Since I'm using a subsample of the data and not all patients are guaranteed to have the same amount of CT images, I'm going to find the patient that has the *most* repeats in this dataset and show all of the images associated with that patient.

# In[ ]:


dup_df = hem_df.pivot_table(index=['patient_id'], aggfunc='size')
dup_df = dup_df[dup_df > 1]

patient_df = hem_df[hem_df["patient_id"] == dup_df.idxmax()]
patient_df = patient_df.sort_values("id")

print("=======PATIENT ID: {}=======".format(patient_df["patient_id"].iloc[0]))

def show_patient_frames(df):
    
    id_list = list(df["id"])

    # Used for subplots but that's been deprecated for larger subset sizes
    num_cols = 3
    num_rows = int(len(id_list) / num_cols)
    
    id_ix = 0
    for row in range(num_rows):
        for col in range(num_cols):
            
            fig = plt.figure(figsize=(8,8))
    
            current_id = id_list[id_ix]
            full_path = train_img_root + "ID_" + current_id + ".dcm"
            dcm_info = pydicom.dcmread(full_path)
            pixel_data = dcm_info.pixel_array

            plt.imshow(pixel_data)

            plt.grid("off")
            plt.axis("off")
#             axes[row, col].set_title("Image ID: {}\nEpidural: {}\nIntraparenchymal: {}\nIntraventricular: {}\nSubdural: {}\nSubarachnoid: {}"
#                  .format(current_id, df.iloc[id_ix, 1], df.iloc[id_ix, 2], df.iloc[id_ix, 3], df.iloc[id_ix, 4], df.iloc[id_ix, 5]))

            plt.title("Image ID: {}\nx: {} y: {} z: {}"
                    .format(current_id, df.iloc[id_ix, 8], df.iloc[id_ix, 9], df.iloc[id_ix, 10]))

            id_ix += 1

    plt.show()
    
show_patient_frames(patient_df)


# # Conclusion
# For now I'll leave it there. We can do further analysis like trying to group by location within the patient but the lack of access to the full dataset makes further analysis a bit difficult. Until I can find a way to work with all of the images in the dataset and split things out by StudyInstanceUID I'll call it here.
