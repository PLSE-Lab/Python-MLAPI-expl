#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle

datapath = '../input/'


# # Lung Radiograph Images
# The folder *stage_1_train_images* contains one image per patient, for a total of 25684 images.

# In[ ]:


# counting the number of files in the image folder
get_ipython().system('ls ../input/stage_1_train_images/ | wc')


# # Bounding Boxes and Target Label Data
# The file *stage_1_train_labels.csv* contains the main training dataset, including the patiend ID, the bounding box coordinates, and the target label [0,1]. There can be multiple rows for the same patient ID, as each row corresponds to one observation (one box) per patient. 
# There are 28989 total boxes, and 25684 unique patient IDs. The negative/positive Target split is roughly 70-30%.

# In[ ]:


df_box = pd.read_csv(datapath+'stage_1_train_labels.csv')
print('Number of rows (unique boxes per patient) in main train dataset:', df_box.shape[0])
print('Number of unique patient IDs:', df_box['patientId'].nunique())
df_box.head(6)


# In[ ]:


df_box.groupby('Target').size().plot.bar()
print(df_box.groupby('Target').size() / df_box.shape[0])


# # Detailed Class Info Data
# The file *stage_1_detailed_class_info.csv* contains detailed information about the positive and negative classes in the training set, and may be used to build more nuanced models. As in the main training dataset, this auxiliary dataset contains 28989 rows and 25684 unique patient IDs. 
# There's 3 classes: 
#     1. Normal (29%)
#     2. No Lung Opacity / Not Normal (40%)
#     3. Lung Opacity (31%)
# The first two classes correspond to Target = 0, whereas the third class correspond to Target = 1.

# In[ ]:


df_aux = pd.read_csv(datapath+'stage_1_detailed_class_info.csv')
print('Number of rows in auxiliary dataset:', df_aux.shape[0])
print('Number of unique patient IDs:', df_aux['patientId'].nunique())
df_aux.head(6)


# In[ ]:


df_aux.groupby('class').size().plot.bar()
print(df_aux.groupby('class').size() / df_aux.shape[0])
assert df_box.loc[df_box['Target']==0].shape[0] == df_aux.loc[df_aux['class'].isin(['Normal',     'No Lung Opacity / Not Normal'])].shape[0], 'Number of negative targets does not match between main and auxiliary dataset.'
assert df_box.loc[df_box['Target']==1].shape[0] == df_aux.loc[df_aux['class'] == 'Lung Opacity'].shape[0],     'Number of positive targets does not match between main and auxiliary dataset.'


# # Merging Main (Boxes) and Auxiliary (Classes) Datasets
# The main and auxiliary datasets do not share a joining keyword column, but it seems obvious that the rows are listed in the exact same order (check the patient Id columns to convince yourself), therefore we can combine the two dataframes by concatenating their columns.

# In[ ]:


assert df_box['patientId'].values.tolist() == df_aux['patientId'].values.tolist(), 'PatientId columns are different.'
df_train = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
df_train.head(6)


# Just for peace of mind, we can check that there is a unique correspondence between Target and class labels.

# In[ ]:


df_train.groupby(['class', 'Target']).size()


# NaN values are only present in the box coordinates columns, as expected.

# In[ ]:


df_train.isnull().any()


# We can also make sure that positive targets are all associated with (non-NaN) box coordinates and viceversa.

# In[ ]:


# when target==1, are any of the box coordinates null? (should all be false)
df_train.loc[df_train['Target']==1, ['x', 'y', 'width', 'height']].isnull().any()


# In[ ]:


# when target==0, are all of the box coordinates null? (should all be true)
df_train.loc[df_train['Target']==0, ['x', 'y', 'width', 'height']].isnull().all()


# # Radiograph Images
# The radiograph images are stored in the folder *stage_1_train_images*. The images are saved in DICOM format (*.dcm*), which includes a header of meta-data and the raw pixel image itself. Images are named using their corrsponding patient ID. Images can be read in and modified using the library [pydicom](https://pydicom.github.io/). The headers of meta-data have been mostly anonymized for patient privacy, but they still contain a bunch of useful information that could be used to improve the classification model. The raw pixel images are stored in 1024x1024 8-bit encoded (=2^8=256 gray-scales) numpy arrays.

# In[ ]:


# sample of image filenames
get_ipython().system('ls -U ../input/stage_1_train_images/ | head -6')


# In[ ]:


# check that there is an image for each unique patient ID
assert sorted(df_train['patientId'].unique().tolist()) == sorted([f[:-4] for f in os.listdir(datapath+'stage_1_train_images/')]),     'Discrepancy between patient IDs and radiograph images.'


# In[ ]:


# have a look at the header meta-data of an image 
pId = df_train['patientId'].sample(1).values[0]    
dcmdata = pydicom.read_file(datapath+'stage_1_train_images/'+pId+'.dcm')
print(dcmdata)


# In[ ]:


# extract the raw pixel image and look at its properties
dcmimg = dcmdata.pixel_array
print(type(dcmimg))
print(dcmimg.dtype)
print(dcmimg.shape)


# In[ ]:


# visualize the corresponding radiograph image
plt.figure(figsize=(20,10))
plt.imshow(dcmimg, cmap=pylab.cm.binary)
plt.axis('off')


# # Define utility functions for visualization
# Below we define a bunch of useful functions to overlay images with boxes and labels.

# In[ ]:


def get_boxes_per_patient(df, pId):
    '''
    Given the dataset and one patient ID, 
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return: 
    array([[x1, y1, width1, height1, class1, target1],
           [x2, y2, width2, height2, class2, target2]])
    '''
    
    boxes = df.loc[df['patientId']==pId][['x', 'y', 'width', 'height', 'class', 'Target']].values
    return boxes


# In[ ]:


def get_dcm_data_per_patient(pId, sample='train'):
    '''
    Given one patient ID and the sample name (train/test), 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(datapath+'stage_1_'+sample+'_images/'+pId+'.dcm')


# In[ ]:


def display_image_per_patient(df, pId, angle=0.0, sample='train'):
    '''
    Given one patient ID and the dataset,
    display the corresponding dicom image with overlaying boxes and class annotation.
    To be implemented: Optionally input the image rotation angle, in case of data augmentation.
    '''
    dcmdata = get_dcm_data_per_patient(pId, sample=sample)
    dcmimg = dcmdata.pixel_array
    boxes = get_boxes_per_patient(df, pId)
    plt.figure(figsize=(14,7))
    plt.imshow(dcmimg, cmap=pylab.cm.binary)
    plt.axis('off')
    
    class_color_dict = {'Normal' : 'green',
                        'No Lung Opacity / Not Normal' : 'orange',
                        'Lung Opacity' : 'red'}

    if len(boxes)>0:
        for box in boxes:
            # extracting individual coordinates and labels
            x, y, w, h, c, t = box 
            # create a rectangle patch
            patch = Rectangle((x,y), w, h, color='red', 
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            # get current axis and draw rectangle
            plt.gca().add_patch(patch)
            
    # add annotation text
    plt.text(10, 50, c, color=class_color_dict[c], size=20, 
             bbox=dict(edgecolor=class_color_dict[c], facecolor='none', alpha=0.5, lw=2))
            


# In[ ]:


## get a sample from each class
samples = df_train.groupby('class').apply(lambda x: x.sample(1))['patientId']

for pId in samples.values:   
    display_image_per_patient(df_train, pId, sample='train')


# # Extract useful meta-data from dicom headers
# We can extract some information from the image headers and add it to the training dataset, so that we can check for possible correlations with the target.

# In[ ]:


def get_metadata_per_patient(pId, attribute, sample='train'):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata = get_dcm_data_per_patient(pId, sample=sample)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value


# In[ ]:


# create list of attributes that we want to extract (manually edited after checking which attributes contained valuable information)
attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_train[a] = df_train['patientId'].apply(lambda x: get_metadata_per_patient(x, a, sample='train'))
# convert patient age from string to numeric
df_train['PatientAge'] = df_train['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_train['PatientAge'] = df_train['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_train.head()


# In[ ]:


# look at age statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby('Target')['PatientAge'].describe()


# In[ ]:


# look at gender statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['PatientSex', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['PatientSex']).size()


# In[ ]:


# look at patient position statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['ViewPosition', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['ViewPosition']).size()


# In[ ]:


# absolute split of view position
df_train.groupby('ViewPosition').size()


# ### **Age and gender - individually - do not seem to be correlated with the target. However, the view position of the radiograph image appears to be really important in terms of target split. AP means Anterior-Posterior, whereas PA means Posterior-Anterior. This [webpage](https://www.med-ed.virginia.edu/courses/rad/cxr/technique3chest.html) explains that "Whenever possible the patient should be imaged in an upright PA position.  AP views are less useful and should be reserved for very ill patients who cannot stand erect". One way to interpret this target unbalance in the ViewPosition variable is that patients that are imaged in an AP position are those that are more ill, and therefore more likely to have contracted pneumonia. Note that the absolute split between AP and PA images is about 50-50, so the above consideration is extremely significant. **

# # Extract Test Images Metadata

# In[ ]:


patientIDs_test = [f[:-4] for f in os.listdir(datapath+'stage_1_test_images/')]
df_test = pd.DataFrame(data={'patientId' : patientIDs_test})


# In[ ]:


attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_test[a] = df_test['patientId'].apply(lambda x: get_metadata_per_patient(x, a, sample='test'))
# convert patient age from string to numeric
df_test['PatientAge'] = df_test['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_test['PatientAge'] = df_test['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_test.head()


# In[ ]:


df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




