#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <table>
#     <tr>
#         <td><img src="https://i.pinimg.com/originals/44/0a/51/440a51fe2f30aadf78e7a0defdbeb672.jpg" width="250"></td>
#         <td><img src="https://cdn.technologyadvice.com/wp-content/uploads/2015/05/dicom-viewers-700x408.jpg" width="350"></td>
#         <td><img src="https://pydicom.github.io/pydicom/stable/_static/pydicom_flat_black.svg" width="150"></td>
#     </tr>
# </table>
# 
# Hey there! I'm going to try to get up and running with a potentially new image format and try to explore some of this dataset. The RSNA Intracranial Hemorrhage dataset was provided to allow Kaggle users to perform classification on images and determine whether or not they contain hemorrhages. Since my background is more medically oriented I'll just be working on some basic analysis and medical image processing techniques, while doing my best to explain any medical background that might be interesting or pertinent as I go through this dataset.
# 
# If you're at all curious about the medical background for this dataset and want to learn more about hemorrhages, feel free to check out [the medical introduction kernel I've created for this competition](https://www.kaggle.com/smit2300/hemorrhage-medical-introduction).
# 
# I'll be focusing more on the actual dataset in this notebook and will assume a little bit of user understanding of the medical background as I move through my exploratory analysis here. Let's jump right in!

# # BASH Dataset Exploration
# 
# The first thing I like to do when I come across a new dataset is get a bit of familiarity with the data directory structure and what sorts of files we'll be dealing with. In a clinical setting this is an extremely important step and has to be done in a scalable way. If someone prepares a dataset and doesn't include navigation tips or instructions to get a clinician started, then a lot of time can be wasted trying to just understand what sort of data is being sent over. These are some of the basic UNIX commands that I run when I come across new medical or other styles of dataset.
# 
# If you're already comfortable with the dataset and with DICOM images then you can go ahead and skip through this section.

# Let's get familiar with the directory layout. I just want to see which files are available
# to us in this dataset.

# In[ ]:


get_ipython().system('printf "==Data Root==\\n"')
get_ipython().system('ls ../input')
get_ipython().system('printf "\\n==Data Structure==\\n"')
get_ipython().system('ls ../input/rsna-intracranial-hemorrhage-detection')

# Alternatively if you've downloaded the dataset locally you could run something like:
# $tree -L 2 ../input


# Rather than use Python's `os` module functions, we can simply use UNIX commands to get a 
# quicker assessment of how many image files are in our training dataset. 
# 
# Remember, the output of the `wc` command is: number of lines | number of words | number of bytes. 
# 
# For our purposes (finding number of files in `ls` output) we're only concerned with the first output.

# In[ ]:


get_ipython().system('printf "\\n==Profiling Training Dataset==\\n"')
get_ipython().system('ls -U ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ | wc')


# Let's see the first 5 image filenames in the training directory. This will be important for
# any sort of regular expressions that might become necessary later in our analysis when
# dealing with filenames that have a specific format.

# In[ ]:


get_ipython().system('printf "\\n==Training Image Filename Examples==\\n"')
get_ipython().system('ls -U ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ | head -5')


# Let's see what the first 5 rows of one of the label csv files looks like as well.

# In[ ]:


get_ipython().system('printf "\\n==Labels CSV Syntax==\\n"')
get_ipython().system('head -5 ../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')


# This is a bit messy. Our CSV file contains 6 rows for each CT image. We'll probably have to cut that down a bit later with some encoding techniques.

# And lastly I want to see what the heck a .dcm file is. I'm assuming a lot of people
# working through this dataset will be seeing this format for the first time so I think
# it'll be a good idea to give a solid foundation to people trying to understand it.

# In[ ]:


get_ipython().system('printf "\\n==DCM File==\\n"')
get_ipython().system('file ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b48228b12.dcm')


# So this image format is very specifically tuned to the display and transfer of medical images--even UNIX knows it! It turns out that Python has a wonderful module called `Pydicom` that makes it possible to go through your whole life without knowing anything about the DICOM image format while still analyzing medical images. If you want a spoiler, you can simply get image data in a numpy array without observing the image data at all. 
# 
# However, with some careful understanding of the file format, it's possible to open up a whole new world of medical data just contained within the images if you are analyzing DICOM images. But more on that later since we can't use the metadata as features for our submission to this competition. For now let's get familiar with this labeling system.

# # EDA
# 
# ## Python Libary Imports

# In[ ]:


import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pydicom
import PIL

from tqdm import tqdm, tqdm_notebook


# ## Data Path Exploration

# In[ ]:


# Get directory names/locations
data_root = os.path.abspath("../input/rsna-intracranial-hemorrhage-detection/")

train_img_root = data_root + "/stage_1_train_images/"
test_img_root  = data_root + "/stage_1_test_images/"

train_labels_path = data_root + "/stage_1_train.csv"
test_labels_path  = data_root + "/stage_1_test.csv"


# ## Image Path Variables

# In[ ]:


# Create list of paths to actual training data
train_img_paths = os.listdir(train_img_root)
test_img_paths  = os.listdir(test_img_root)

# Dataset size
num_train = len(train_img_paths)
num_test  = len(test_img_paths)

print("Train dataset consists of {} images".format(num_train))
print("Test  dataset consists of {} images".format(num_test))


# ## Large Dataset Importing
# 
# Since we're looking at a relatively large dataset I'm going to use a method to load in our CSV to memory that might be a bit of overkill. I'm using a method described by Kaggle user [szelee](https://www.kaggle.com/szelee) in his excellent [notebook on NYC Taxi rides](https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows). 
# 
# The main method I'm going to be borrowing from that notebook is the changing of dataset import datatypes. Since we're only importing the labels this might not be necessary, but even datasets with only two columns can end up using up a significant amount of RAM.

# In[ ]:


df_tmp = pd.read_csv(
    train_labels_path,
    nrows=5
)

columns = list(df_tmp.columns)

print("\nFeatures in training labels:")
for column in columns:
    print(column)

print("\nDataFrame Datatype Information:")
print(df_tmp.info())


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def create_efficient_df(data_path):\n    \n    # Define the datatypes we\'re going to use\n    final_types = {\n        "ID": "str",\n        "Label": "float32"\n    }\n    features = list(final_types.keys())\n    \n    # Use chunks to import the data so that less efficient machines can only use a \n    # specific amount of chunks on import\n    df_list = []\n\n    chunksize = 1_000_000\n\n    for df_chunk in pd.read_csv(data_path, dtype=final_types, chunksize=chunksize): \n        df_list.append(df_chunk)\n        \n    df = pd.concat(df_list)\n    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]\n\n    del df_list\n\n    return df\n\ntrain_labels_df = create_efficient_df(train_labels_path)\n# test_labels_df  = create_efficient_df(test_labels_path)')


# In[ ]:


train_labels_df.info()


# About 80MB of RAM used even on our optimized import! I really like the attention to detail that this process brings regardless of whether or not it's necessary. As an engineer it's an important element of the job being able to know the exact certainty that's necessary for a given task and for data scientists this certainty comes in the form of datatype precision. 
# 
# Pandas is a wonderful tool, but if you're not careful it can be a bit bloated. It only takes a couple of minutes and some technical care and know-how to really prune down an otherwise unwieldy data structure in your machine's memory. 

# In[ ]:


# Syntax = Which image + hemorrhage type, Probability image contains that hemorrhage type
train_labels_df.head(10)


# # Label Encoding
# This labeling dataset is pretty large--I would argue about 6x larger than it needs to be. Having a new row for each prediction seems like a waste of vertical scaling space to me. Let's try to clean it up in a way that will be easier for future machine learning/deep learning models to work with futher down the line.
# 
# I would like the dataframe to have one row per image and within that row contain the certainty the image contains each type of hemorrhage as a column of that row. 
# 
# We'll encode a new dataframe to use the following names corresponding to each hemorrhage type:
#  * type_0: epidural
#  * type_1: intraparenchymal
#  * type_2: intraventricular
#  * type_3: subarachnoid
#  * type_4: subdural
#  * type_5: any

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
new_labels_df = pd.DataFrame(data=empty_array, columns=new_cols)

# Fill in the ID of each image
new_labels_df["id"] = list(train_labels_df.iloc[::len(hem_types)]["ID"].str.split(pat="_").str[1])
    
# Fill in the categorical columns of each image
for hem_ix, hem_col in enumerate(list(new_labels_df)[1:]):
    new_labels_df[hem_col] = list(train_labels_df.iloc[hem_ix::len(hem_types), 1])
                        
new_labels_df.sample(10)


# That's much better. Now instead of a dataset containing 4 million rows, we have one that's down to 674K and each row has a richer set of information.

# # DICOM Introduction
# 
# <img src="https://www.lifewire.com/thmb/n1mn_MyOKGr1VPEqsZ1xPfYF7UQ=/768x0/filters:no_upscale():max_bytes(150000):strip_icc()/brain-scan-peter-dazeley-photographers-choice-getty-images-56e09ed65f9b5854a9f855fc.jpg" width="500">
# 
# The DICOM image format is a beautiful thing once you get to know it. As a data scientist or imaging scientist, it can combine information from two worlds very nicely. This format gives context to images in a way that is crucial for physicians to know where within a scan they are looking as well as data for an engineer to know where in a scan they are predicting.

# In[ ]:


random_ix = random.randint(0, len(train_img_paths))
random_path = train_img_root + train_img_paths[random_ix]

dcm_info = pydicom.dcmread(random_path)
print("===DICOM MEDICAL INFO===")
print(dcm_info)

pixel_data = dcm_info.pixel_array
print("\n===IMAGE PIXEL INFO===")
print("Image dimensions: {}".format(pixel_data.shape))
print(np.max(pixel_data))
print(np.min(pixel_data))
print(np.median(pixel_data))

plt.figure(figsize=(10,10))
sns.distplot(pixel_data.flatten())
plt.title("Pixel Brightness Distribution for DICOM Image")


# ### Information DICOM Tells Us
# 
# One assumption we might have made without knowing how DICOM works is that each image ID in the .dcm filename tells us which patient the image belongs to. However, when we inspect the DICOM file, we get a unique Patient ID field that is different than the image ID. We can use this information to build out a full profile for a given patient. This can set up data structures that contain entire scans for a patient and allow for things like 3D reconstruction of a scan, more accurate and localized hemorrhage diagnoses, and generally a more contextualized image. 

# # Viewing Some DICOM Images

# In[ ]:


# Function to show a random image containing a specific type of hemorrhage
# We can set the threshold to be lower or higher as well. 
def show_random_sample(hem_choice, thresh):
    
    types = new_labels_df.columns[1:]
    chosen_type = types[hem_choice]
    
    print("Displaying image with >= %.2f%% chance of containing an _%s_ hemmorhage..." % (thresh*100, chosen_type))

    filtered_df = new_labels_df[new_labels_df[chosen_type] > thresh]
    
    random_ix = random.randint(0, filtered_df.shape[0])
    
    target_record = filtered_df.iloc[random_ix, :]
    target_id = target_record[0]
    image_path = train_img_root + "ID_" + target_id + ".dcm"
    
    print("Opening {}...".format(image_path))
    
    print(target_record)
    
    dcm_img = pydicom.dcmread(image_path)
    plt.imshow(dcm_img.pixel_array)
    
    plt.grid("off")
    plt.axis("off")
    plt.title("Image of Patient with {} Hemorrhage".format(hem_types[hem_choice].title()))
    
    plt.show()


# Let's see a couple of different types of hemorrhages with high probabilities.

# In[ ]:


for type in range(6):
    show_random_sample(type, 0.8)


# In[ ]:


def display_by_id(patient_id):

    image_path = train_img_root + "ID_" + patient_id + ".dcm"
    
    print("Opening {}...".format(image_path))
        
    dcm_img = pydicom.dcmread(image_path)
    plt.imshow(dcm_img.pixel_array)
    
    plt.grid("off")
    plt.axis("off")
    plt.title("Image of Patient {}".format(patient_id))


# In[ ]:


display_by_id("4e16848f1")


# # Conclusion
# 
# That's it for now! My next notebook is going to explore splitting out DICOM images by patient and evaluating hemorrhages on a per-patient basis. I hope this was a useful introduction to this dataset as well as to the DICOM image format!
