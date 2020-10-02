#!/usr/bin/env python
# coding: utf-8

# <H1>Dicom Dataset EDA</H1>
# 
# The images in this contest are supplied in a number of ways including jpegs, tfRecords, and Dicom datasets. But as this contest is posed as a medical imaging problem, one has to wonder if it wouldn't be best to use the medical images themselves.  I have worked with dicom images once before as they were the only data provided for the 2017 Data Science Bowl contest on Kaggle.
# 
# The EDA below is a brief exploration of these images and using them with the pydicom library which you can install easily with "pip install pydicom"

# In[ ]:


get_ipython().system('pip install p_tqdm')


# In[ ]:


# ----------------------------------------
# imports
# ----------------------------------------
import os
import pydicom
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
# uncomment if you want to run the dicom data collection code
from multiprocessing import pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from p_tqdm import p_map #pip install p_tqdm
import pandas as pd
import numpy as np


# In[ ]:


# ----------------------------------------
# constants
# ----------------------------------------
root_path = '/kaggle/input/siim-isic-melanoma-classification'
data_path = root_path
output_path = '/kaggle/working'


# <H2>Dicom Metadata Exploration</H2>
# Dicom datasets come with a variety of metadata items embedded within them.  Dicom is a standard and each metadata item has a specific meaning an purpose.  Here I'll summarize a few, but you can look up anything you feel you need to know for this challenge at <A HREF="https://dicom.innolitics.com/ciods/segmentation/general-image/00080008">this website.</A>  As you check this out note the right hand column which is of the form (xxxx, xxxx).  Make sure to use these references to make sure you are looking at the right definition as the text descriptions are reused across different image types.
# <br>
# <br>
# The <B>Image Type</B> shown here indicates that this is an image whose pixel values have been derived in some manner from the pixel value of one or more other images.  Also, this was not created as a direct result of a patient examination. Instead this image is a SECONDARY image; an image created after the initial patient examination.
# <br>
# <br>
# The <B>Modality</B> is a code indicating the equipment used to acquire the image.  In this case, XC means that the image was acquired by external camera.
# <br>
# <br>
# One nuance to note is the interaction between <B>Samples Per Pixel</B> and <B>Photometric Interpretation</B>.  Samples Per Pixel is the number of separate planes in this image.  So no real surprise its a 3 plane image (think RGB, YUV, YCbCr).  But note that the Photometric Interpretation is  YBR_FULL_422 which means that this image is a YCbCr representation but Cb and Cr values are sampled horizontally at half the Y rate and as a result there are half as many Cb and Cr values as Y values.  Therefore, the Samples per Pixel describes the nominal number of channels (i.e., 3), and does not reflect that two chrominance samples are shared between four luminance samples.
# <br>
# <br>
# Many of the data elements are self explanatory.  So <B>Rows</B> and <B>Columns</B> are just image dimensions, and of course the image itself is called <B>Pixel Data</B>.
# <br>
# <br>
# I'll leave the rest for you to research yourself.  If you find something useful or helpful to us all, please post it!

# In[ ]:


# ------------------------------------------------------
# get a list of all of the training dicom files
# ------------------------------------------------------
dcom_list = [f for f in os.listdir(f'{data_path}/train')]
dcom_test_list = [f for f in os.listdir(f'{data_path}/test')]


# In[ ]:


# ------------------------------------------------------
# read in a dicom image and all of its metadata
# ------------------------------------------------------
dataset = pydicom.dcmread(f'{data_path}/train/{dcom_list[0]}')


# In[ ]:


# ------------------------------------------------------
# print out a full listing
# ------------------------------------------------------
for element in dataset:
    print(element)


# <H2>Metadata Statistics</H2>
# I know that the competition organizers supplied us with a CSV to go along with the JPEG images. But this file only includes a few of the items that are available to us in the Dicom images.  Lets have a look at a broader set of metadata elements.  
# <br>
# Note that the following code block may take a little time as we have to open every Dicom dataset, extract the data elements we want and build a list.  I've saved out a csv if you want to skip running this.  That said, I did parallelize so it runs as quickly as possible.  Your mileage may vary.

# In[ ]:


# -------------------------------------------------------------------------------------------
# helper function to open a dicom image, extract some meta data and return a list of values
# this works with the thread pooler below to reduce wait time
# to get both test and train you will have to switch the data_type argument to the desired
# folder
# -------------------------------------------------------------------------------------------
def collect_dcom_data(dcom_file, data_path=data_path):
    ds = pydicom.dcmread(f'{data_path}/train/{dcom_file}')
    return [ds.PatientID, ds.PatientSex, ds.PatientAge, ds.BodyPartExamined, ds.InstitutionName, 
            ds.ImageType, ds.Modality, ds.PhotometricInterpretation, ds.Rows, ds.Columns]
def collect_dcom_data_test(dcom_file, data_path=data_path):
    ds = pydicom.dcmread(f'{data_path}/test/{dcom_file}')
    return [ds.PatientID, ds.PatientSex, ds.PatientAge, ds.BodyPartExamined, ds.InstitutionName, 
            ds.ImageType, ds.Modality, ds.PhotometricInterpretation, ds.Rows, ds.Columns]


# In[ ]:


# -------------------------------------------------------------------------------------------
# Multithreaded extract of meta data
# I was using a GCP instance which only provides vCPUs (one thread per CPU)
# your mileage will vary
#
# I've saved csv's of the results so you just skip this part if you want
# -------------------------------------------------------------------------------------------
pool = ThreadPool(cpu_count())  
meta_data = p_map(collect_dcom_data, dcom_list)   
df = pd.DataFrame(meta_data, columns=['PatientID', 'PatientSex', 'PatientAge', 'BodyPartExamined', 'InstitutionName', 
                                      'ImageType', 'Modality', 'PhotometricInterpretation', 
                                      'Rows', 'Columns'])
df.to_csv(output_path + '/' + 'metadata.csv', index=False)


meta_data = p_map(collect_dcom_data_test, dcom_test_list)   
df_test = pd.DataFrame(meta_data, columns=['PatientID', 'PatientSex', 'PatientAge', 'BodyPartExamined', 'InstitutionName',
                                           'ImageType', 'Modality', 'PhotometricInterpretation', 
                                           'Rows', 'Columns'])
df_test.to_csv(output_path + '/' + 'metadata_test.csv', index=False)


# In[ ]:


df = pd.read_csv(output_path + '/' + 'metadata.csv')
df_test = pd.read_csv(output_path + '/' + 'metadata_test.csv')


# <H3>Modality</H3>
# All of the images are "XC" meaning they were all taken via an external camera.  This is true of both the train and test data.

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# create the first chart
df_mods = df.Modality.value_counts()
ax1 = df_mods.plot.pie(ax=ax1)

# create the second chart
df_mods = df_test.Modality.value_counts()
ax2 = df_mods.plot.pie(ax=ax2)


# <H3>ImageType</H3>
# All of the images contain pixel values have been derived in some manner from the pixel value of one or more other images.  This is also true for both train and test.

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# create the first chart
df_type = df.ImageType.value_counts()
ax1 = df_type.plot.pie(ax=ax1, labels=None)

# create the second chart
df_type = df_test.ImageType.value_counts()
ax2 = df_type.plot.pie(ax=ax2, labels=None)


# <H3>Image Width</H3>
# There are 88 combinations of height and width for these images.  The chart below shows the top 10 width and the related image counts with those widths.  The distribution between train and test are similar, but given the resizes that will be necessary, there are some differences to note.

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# create the first chart
df_cols = df.Columns.value_counts()
df_cols[:10].plot.bar(ax=ax1)

# create the second chart
df_cols = df_test.Columns.value_counts()
df_cols[:10].plot.bar(ax=ax2)

plt.show()


# <H3>Image Height</H3>
# The chart below shows the top 10 heights and the related image counts.  Same resize concerns are relevant here.

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# create the first chart
df_rows = df.Rows.value_counts()
df_rows[:10].plot.bar(ax=ax1)

# create the second chart
df_rows = df_test.Rows.value_counts()
df_rows[:10].plot.bar(ax=ax2)

plt.show()


# <H3>Gender Distribution</H3>
# There are slightly more men than women in the training data and even more imbalance in the test set.  There are also a few records for which gender is not known in the training set. 

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# charts
ax1 = df.PatientSex.value_counts().plot.pie(ax=ax1)
ax2 = df_test.PatientSex.value_counts().plot.pie(ax=ax2)

plt.show()


# <H3>Age Distribution</H3>
# The age distributions are also reasonably similar although the test set is a little notchy towards the right tail. 

# In[ ]:


# set up the fig
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_title('Train')
ax2.set_title('Test')

# create the first chart
ages = df.PatientAge.unique()
ages.sort()
ax1 = df.PatientAge.value_counts().reindex(ages.tolist()).plot.bar(ax=ax1)

# create the second chart
ages = df_test.PatientAge.unique()
ages.sort()
ax2 = df_test.PatientAge.value_counts().reindex(ages.tolist()).plot.bar(ax=ax2)

plt.show()


# <H2>Dicom Images</H2>
# As noted above, the Dicom images are natively a modified version of the YCrCb color space. Lets have a look at some of the differences between the JPEGs and the Dicom images.

# In[ ]:


# you can display a dcom image easily with matplotlib
plt.imshow(dataset.pixel_array)
plt.show()


# In[ ]:


# get a list of all of the training jpeg files
jpeg_list = [f for f in os.listdir(f'{data_path}/jpeg/train')]


# In[ ]:


# -----------------------------------------------
# Displays differences for a single image
# -----------------------------------------------
def display_dif_compare(image_name, nrows=4, ncols=3, figsize=15):
    
    # clean file name and open image in variety of styles
    file_name, _ = os.path.splitext(image_name)
    ds = pydicom.dcmread(f'{data_path}/train/{file_name}.dcm')

    dcom_image = ds.pixel_array
    jpeg = cv2.imread(f'{data_path}/jpeg/train/{file_name}.jpg', cv2.IMREAD_UNCHANGED)
    #jpeg_rgb = jpeg
    jpeg_rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
    jpeg_ycc = cv2.cvtColor(jpeg, cv2.COLOR_BGR2YCR_CB)
    dcom_rgb = cv2.cvtColor(dcom_image, cv2.COLOR_YCR_CB2RGB)

    jpeg_ycc_pil = Image.open(f'{data_path}/jpeg/train/{file_name}.jpg')
    jpeg_ycc_pil.draft('YCbCr', None)
    jpeg_ycc_pil.load() 

    # plot the figures (apologies for the brute force inelegance of this block)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize, figsize))

    axarr[0, 0].set_title('Raw Dicom')
    axarr[0, 0].imshow(dcom_image)
    axarr[0, 1].set_title('JPEG Raw via cv2')
    axarr[0, 1].imshow(jpeg_rgb)
    axarr[0, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_rgb)//1000000) + ' x 10^5')
    axarr[0, 2].imshow(abs(dcom_image - jpeg_rgb))
    axarr[0, 0].set(ylabel = file_name)
    
    axarr[1, 0].set_title('Raw Dicom')
    axarr[1, 0].imshow(dcom_image)
    axarr[1, 1].set_title('JPEG Raw->YCrCb via cv2')
    axarr[1, 1].imshow(jpeg_ycc)
    axarr[1, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_ycc)//1000000) + ' x 10^5')
    axarr[1, 2].imshow(abs(dcom_image - jpeg_ycc))
    axarr[1, 0].set(ylabel = file_name)
    
    axarr[2, 0].set_title('Dicom Raw->RGB via cv2')
    axarr[2, 0].imshow(dcom_rgb)
    axarr[2, 1].set_title('JPEG Raw via cv2')
    axarr[2, 1].imshow(jpeg_rgb)
    axarr[2, 2].set_title('Dif: ' + str(np.sum(dcom_rgb - jpeg_rgb)//1000000) + ' x 10^5')
    axarr[2, 2].imshow(abs(dcom_rgb - jpeg_rgb))
    axarr[2, 0].set(ylabel = file_name)
    
    axarr[3, 0].set_title('Dicom Raw')
    axarr[3, 0].imshow(dcom_image)
    axarr[3, 1].set_title('JPEG Raw-YCrCb via PIL "draft"')
    axarr[3, 1].imshow(jpeg_ycc_pil)
    axarr[3, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_ycc_pil)//1000000) + ' x 10^5')
    axarr[3, 2].imshow(abs(dcom_image - jpeg_ycc_pil))
    axarr[3, 0].set(ylabel = file_name)


# <H3>Image Handling Will Be Important</H3>
# Here we compare a single image between the Dicom pixel array and the jpeg image.  Depending upon how you handle the opening and conversion of each image format, you will get various differences that may effect your model.  Recall that the Raw Dicom image is a modified YCbCr image, while the JPEG when opened by cv2 is opened in BGR format (not RGB).  Note also that if you open the image via cv2 in the normal imread fashion.  The image will be manipulated through a number of steps (see <A HREF="https://www.graphicsmill.com/docs/gm5/UnderstandingofJPEGEncodingParameters.htm">here</A>. 

# In[ ]:


image_name = np.random.choice(jpeg_list)
display_dif_compare(image_name)


# <H3>RGB Differences</H3>
# Here you can see the differences between the two images in each color channel.  Its kind of hard to imagine this will have no impact on model results.

# In[ ]:


file_name, _ = os.path.splitext(image_name)
ds = pydicom.dcmread(f'{data_path}/train/{file_name}.dcm')
image1 = cv2.cvtColor(ds.pixel_array, cv2.COLOR_YCR_CB2RGB)
image2 = cv2.imread(f'{data_path}/jpeg/train/{image_name}', cv2.IMREAD_UNCHANGED)

# tuple to select colors of each channel line
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

plt.figure(figsize=(20,5))

# create the histogram plot, with three lines, one for
# each color for each image
for channel_id, c in zip(channel_ids, colors):
    histogram1, bin_edges = np.histogram(image1[:, :, channel_id], bins=256, range=(0, 256))

for channel_id, c in zip(channel_ids, colors):
    histogram2, bin_edges = np.histogram(image2[:, :, channel_id], bins=256, range=(0, 256))


# plot the reds for each image
ax1 = plt.subplot(1,3,1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='r', label='Dicom-RGB via cv2')
plt.plot(bin_edges[0:-1], histogram2, color='r', linestyle='dashed', label='JPEG Raw via cv2')
plt.legend()
plt.xlabel("Color value")
plt.ylabel("Pixels")

# plot the greens for each image
ax2 = plt.subplot(1,3,2, sharey=ax1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='g', label='Dicom-RGB via cv2')
plt.plot(bin_edges[0:-1], histogram2, color='g', linestyle='dashed', label='JPEG Raw via cv2')
plt.legend()
plt.xlabel("Color value")
plt.ylabel("Pixels")

# plot the greens for each image
ax3 = plt.subplot(1,3,3, sharey=ax1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='b', label='Dicom-RGB via cv2')
plt.plot(bin_edges[0:-1], histogram2, color='b', linestyle='dashed', label='JPEG Raw via cv2')
plt.legend()
plt.xlabel("Color value")
plt.ylabel("Pixels")


# <H3>Luminance and Chrominance Differences</H3>
# Here you can see the differences between the two images in the Y, Cb, and Cr channels.

# In[ ]:


image1 = ds.pixel_array
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YCR_CB)

# tuple to select colors of each channel line
colors = ("y", "Cb", "Cr")
channel_ids = (0, 1, 2)

plt.figure(figsize=(20,5))

# create the histogram plot, with three lines, one for
# each color for each image
for channel_id, c in zip(channel_ids, colors):
    histogram1, bin_edges = np.histogram(image1[:, :, channel_id], bins=256, range=(0, 256))

for channel_id, c in zip(channel_ids, colors):
    histogram2, bin_edges = np.histogram(image2[:, :, channel_id], bins=256, range=(0, 256))


# plot the reds for each image
ax1 = plt.subplot(1,3,1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='y', label='Dicom Raw')
plt.plot(bin_edges[0:-1], histogram2, color='y', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')
plt.legend()
plt.xlabel("Luminance")
plt.ylabel("Pixels")

# plot the greens for each image
ax2 = plt.subplot(1,3,2, sharey=ax1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='b', label='Dicom Raw')
plt.plot(bin_edges[0:-1], histogram2, color='b', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')
plt.legend()
plt.xlabel("Cb")
plt.ylabel("Pixels")

# plot the greens for each image
ax3 = plt.subplot(1,3,3, sharey=ax1)
plt.xlim([0, 256])
plt.plot(bin_edges[0:-1], histogram1, color='r', label='Dicom Raw')
plt.plot(bin_edges[0:-1], histogram2, color='r', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')
plt.legend()
plt.xlabel("Cr")
plt.ylabel("Pixels")


# <H3>Conclusion</H3>
# Hopefully you find this information on the Dicom format and comparisons to the JPEG format helpful.  If you do, please give me an upvote!  Thanks.
