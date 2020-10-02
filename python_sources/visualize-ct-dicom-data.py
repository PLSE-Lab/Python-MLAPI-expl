#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Visualize CT DICOM Data</font></center></h1>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a> 
#     - <a href='#31'>Read overview data</a> 
#     - <a href='#32'>Read TIFF data</a> 
#     - <a href='#33'>Read DICOM data</a>  
# - <a href='#4'>Data exploration</a>
#     - <a href='#41'>Check data consistency</a> 
#     - <a href='#42'>Show TIFF images</a> 
#     - <a href='#43'>Show DICOM data</a> 
# - <a href='#5'>Conclusions</a>
# - <a href='#6'>References</a>

# # <a id="1">Introduction</a>
# 
# ## Overview  
# 
# The dataset is designed to allow for different methods to be tested for examining the trends in CT image data associated with using contrast and patient age. The basic idea is to identify image textures, statistical patterns and features correlating strongly with these traits and possibly build simple tools for automatically classifying these images when they have been misclassified (or finding outliers which could be suspicious cases, bad measurements, or poorly calibrated machines)
# 
# ## Data
# The data are a tiny subset of images from the cancer imaging archive. They consist of the middle slice of all CT images taken where valid age, modality, and contrast tags could be found.   TCIA Archive Link - [https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD)     
# The images data is provided both in DICOM and TIFF formats. The images data files are named with a naming convention allowing us to identify some meta-data about the images.
# 
# 
# 
# ## DICOM format
# 
# **Digital Imaging and Communications in Medicine** (**DICOM**) is the accepted standard for the communication and management of medical imaging information.  **DICOM** is used for archiving and transmitting medical images. It enables the integration of medical imaging devices (radiological scanners), servers, network hardware and **Picture Archiving and Communication Systems** (**PACS**). The standard was widely adopted by hospitals and research centers and is steadly advancing as well toward small practice and cliniques.     
# 
# 
# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 

# # <a id="2">Load packages</a>
# 
# We will load the packages for showing tiff images and dicom data.   
# 
# For dicom data, we are loading the **dicom** package.   
# 

# In[ ]:


IS_LOCAL = False
import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
if(IS_LOCAL):
    import pydicom as dicom
else:
    import dicom
import os


# Parameters setting and files list.

# In[ ]:


if(IS_LOCAL):
    PATH="../input/siim-medical-image/"
else:
    PATH="../input/"
print(os.listdir(PATH))


# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 
# 
# # <a id="3">Read the data</a>
# 
# 
# ## <a id="31">Read overview data</a>

# In[ ]:


data_df = pd.read_csv(os.path.join(PATH,"overview.csv"))


# In[ ]:


print("CT Medical images -  rows:",data_df.shape[0]," columns:", data_df.shape[1])


# In[ ]:


data_df.head()


# ## <a id="32">Read TIFF data</a>  
# 

# In[ ]:


print("Number of TIFF images:", len(os.listdir(os.path.join(PATH,"tiff_images"))))


# In[ ]:


tiff_data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH+'tiff_images/*.tif')])


# ### Process TIFF data   
# 
# We define a function to process data.   
# We extract file, ID, age, contrast, modality information from path info.

# In[ ]:


def process_data(path):
    data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH+path)])
    data['file'] = data['path'].map(os.path.basename)
    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data


# In[ ]:


tiff_data = process_data('tiff_images/*.tif')


# ### Check TIFF data
# 
# Let's check the TIFF data, after we extracted the meta info from the file name.

# In[ ]:


tiff_data.head(10)


# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 
# 
# ## <a id="33">Read DICOM data</a>
# 
# 
# We repeat the same processing for the **DICOM** data.

# In[ ]:


print("Number of DICOM files:", len(os.listdir(PATH+"dicom_dir")))


# ### Process DICOM data

# In[ ]:


dicom_data = process_data('dicom_dir/*.dcm')


# ### Check DICOM data

# In[ ]:


dicom_data.head(10)


# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>  
# 
# 
# # <a id="4">Data exploration</a>

# ## <a id="41">Check data consistency</a>
# 
# Let's verify if the content in overview.csv is consistent with the data in tiff_images folder.

# In[ ]:


def countplot_comparison(feature):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview data")
    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff files data")
    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom files data")
    plt.show()


# In[ ]:


countplot_comparison('Contrast')


# In[ ]:


countplot_comparison('Age')


# The values in the 3 data sources are consistent.   
# 
# 
# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>

# ## <a id="42">Show TIFF images</a>
# 
# We will show a subsample of 16 images from the total of 100 images.     
# We will select the first 16 images from the data set.   
# We will use grayscale.   
# We define here a generic function to represent both TIFF images and DICOM images.
# For each file format we use a different processing.

# In[ ]:


def show_images(data, dim=16, imtype='TIFF'):
    img_data = list(data[:dim].T.to_dict().values())
    f, ax = plt.subplots(4,4, figsize=(16,20))
    for i,data_row in enumerate(img_data):
        if(imtype=='TIFF'): 
            data_row_img = imread(data_row['path'])
        elif(imtype=='DICOM'):
            data_row_img = dicom.read_file(data_row['path'])
        if(imtype=='TIFF'):
            ax[i//4, i%4].matshow(data_row_img,cmap='gray')
        elif(imtype=='DICOM'):
            ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title('Modality: {Modality} Age: {Age}\nSlice: {ID} Contrast: {Contrast}'.format(**data_row))
    plt.show()


# We apply the function to show TIFF images.

# In[ ]:


show_images(tiff_data,16,'TIFF')


# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 
# 
# ## <a id="43">Show DICOM data</a>
# 
# We will show a subsample of 16 images from the total of 100 images.   
# We will use grayscale.   
# Ideally, if the **DICOM** images would be a set of slices from a single examen, they could be aggregated using a function like the one shown here: extract_voxel_data - which read the **DICOM** slices (each in a separate file) and aggregate the image data in a **3D voxel tensor**. This will not be the case here, because we are storing slices from different patients and exams (one slice / exam / patient).
# 
# The following code snapshot shows how tipically a DICOM 2D image subset is used to create a 3D scene.

# >     # extract voxel data  
# >     def extract_voxel_data(list_of_dicom_files):  
# >         datasets = [dicom.read_file(f) for f in list_of_dicom_files]  
# >          try:  
# >              voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)  
# >          except dicom_numpy.DicomImportException as e:  
# >          # invalid DICOM data  
# >              raise  
# >          return voxel_ndarray  

# Here we show a subset of 16 images.

# In[ ]:


show_images(dicom_data,16,'DICOM')


# ### More about DICOM data
# 
# A DICOM file containg much more information than the image itself that we represented. Let's glimpse, for one of the DICOM files, this information. We will read the first dicom file only and show this information.  
# We use **dicom** package.

# In[ ]:


dicom_file_path = list(dicom_data[:1].T.to_dict().values())[0]['path']
dicom_file_dataset = dicom.read_file(dicom_file_path)
dicom_file_dataset


# We can extract various fields from the DICOM FileDataset. Here are few examples:  
# * Modality  
# * Manufacturer
# * Patient Age  
# * Patient Sex
# * Patient Name  
# * Patient ID
# 
# 

# In[ ]:


print("Modality: {}\nManufacturer: {}\nPatient Age: {}\nPatient Sex: {}\nPatient Name: {}\nPatient ID: {}".format(
    dicom_file_dataset.Modality, 
    dicom_file_dataset.Manufacturer,
    dicom_file_dataset.PatientAge,
    dicom_file_dataset.PatientSex,
    dicom_file_dataset.PatientName,
    dicom_file_dataset.PatientID))


# Some of the information are anonymized (like Name and ID), which is common standard for public medical data.   
# 
# We will modify the visualization function, to show parameters from the DICOM data instead of the parameters extracted from the image name.  
# 
# 

# In[ ]:


def show_dicom_images(data):
    img_data = list(data[:16].T.to_dict().values())
    f, ax = plt.subplots(4,4, figsize=(16,20))
    for i,data_row in enumerate(img_data):

        data_row_img = dicom.read_file(data_row['path'])
        modality = data_row_img.Modality
        age = data_row_img.PatientAge
        
        ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title('Modality: {} Age: {}\nSlice: {} Contrast: {}'.format(
         modality, age, data_row['ID'], data_row['Contrast']))
    plt.show()


# In[ ]:


show_dicom_images(dicom_data)


# In[ ]:



<a href="#0"><font size="1" color="red">Go to top</font></a>


# # <a id="5">Conclusion</a>
# 
# We demonstrated how we can load and show **TIFF** images.   
# As well, using **dicom** and **dicom-numpy** packages, we demonstrated how to read and visualize **DICOM** data.   
# We also explored preliminary the content of a **DICOM** data file and modified the visualization function to use (partially) **DICOM** data for the image attributes.  
# 
# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 

# # <a id="6">References</a>
# 
# [1] <a href="https://www.kaggle.com/kmader">Kevin Mader</a>,  <a href="https://www.kaggle.com/kmader/show-the-data-in-the-zip-file">Show the data in the Zip File</a>    
# [2] <a href="https://www.kaggle.com/byrachonok">Vitaly Byrachonok</a>,  <a href="https://www.kaggle.com/byrachonok/study-ct-medical-images">Study CT Medical Images</a>    
# [3] Python package for processing DICOM data, dicom-numpy, https://dicom-numpy.readthedocs.io     
# [4] Viewing DICOM images in Python, https://pydicom.github.io/pydicom/stable/viewing_images.html     
# [5] DICOM format, https://en.wikipedia.org/wiki/DICOM    
# 
# 
# 
# <a href="#0"><font size="1" color="red">Go to top</font></a>
# 
# 
# 
