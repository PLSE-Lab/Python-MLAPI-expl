#!/usr/bin/env python
# coding: utf-8

# Kaggle has put forward another image challenge! I love these because convolutional networks are pretty nifty, and they are the perfect too for this job. Let's begin by getting to know the data and what format it is stored in! to do this we are going to need to import some libraries.
# 
# **A wild Image Format appeared!** This is the first time i have dealt with DICOM images personally. I was vaguely aware of their existence working as a technical analyst for an EHR system however I've never had the opportunity to get down and dirty with them! Another reason I love Kaggle!
# 
# The new library I am referring to is the pydicom library. Reading through the documentation it appears that a DCM image is actually an archive with a lot of interesting meta data and an image. The meta data in these images isnt quite as interesting as i hoped since the data has been deidentified however its still awesome to have such a fun storage format to play with.
# 
# Lets load some DICOM files, view some metadata and plot the images.

# In[ ]:


import pydicom
import os
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm


# No we can define a function to view some interesting metadata fields, this is taken almost directly from the [pydicom website](https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html), I have just added a few fields I found interesting and turned it into a function since I beleive that code is easier to manage like this, even in Jupyter Notebooks.

# In[ ]:


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


# We will also need a function to plot the images (we will be using matplotlib.pyplot for this). I've included the option to control the figure size, you can fork this Notebook and adjust that default argument for the function if you have a smaller/bigger screen.

# In[ ]:


def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


# Now we have all our ground work in place, lets look at what is required to read and load a DICOM image from disk! We will only visualise a few images here.

# In[ ]:


i = 1
num_to_plot = 5
for file_name in os.listdir('../input/stage_1_train_images/'):
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)
    
    if i >= num_to_plot:
        break
    
    i += 1


# Wow! These images are fairly large! they are 1024x1024 pixels, and they look wonderful when plotted. This is going to be a spectacular project/competition to work on! The bone colour map is a beutiful touch, they give the plots that authentic X-ray feeling.

# Looking at the images we can see that the Female and Male bodies actually do differ in shape and size a little bit. Lets do some EDA on the gender and ages of clients in this dataset and find out what sort of distributions we are dealing with!

# In[ ]:


train_demo_df = pd.DataFrame()
ids = []
ages = []
sexs = []
img_avg_lums = []
img_max_lums = []
img_min_lums = []

from multiprocessing.pool import Pool, ThreadPool

pool = ThreadPool(4)

def process_image(dataset):
    _id = dataset.PatientID
    _age = dataset.PatientAge
    _sex = dataset.PatientSex
    _mean = np.mean(dataset.pixel_array)
    _min = np.max(dataset.pixel_array)
    _max = np.min(dataset.pixel_array)
    return _id, _age, _sex, _min, _max, _mean

responses = []
for file_name in tqdm(os.listdir('../input/stage_1_train_images/')):
    
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)

    responses.append(pool.apply_async(process_image, (dataset,)))


pool.close()
pool.join()


# In[ ]:


for response in tqdm(responses):
    _id, _age, _sex, _min, _max, _mean = response.get()
    ids.append(_id)
    ages.append(_age)
    sexs.append(_sex)
    img_min_lums.append(_min)
    img_max_lums.append(_max)
    img_avg_lums.append(_mean)


train_demo_df['patientId'] = pd.Series(ids)
train_demo_df['patientAge'] = pd.Series(ages, dtype='int')
train_demo_df['patientSex'] = pd.Series(sexs)

train_demo_df['imageMin'] = pd.Series(img_max_lums)
train_demo_df['imageMax'] = pd.Series(img_min_lums)
train_demo_df['imageMean'] = pd.Series(img_avg_lums)

sex_map = {'F': 0, 'M': 1}
train_demo_df['patientSex'] = train_demo_df['patientSex'].replace(sex_map).astype('int')


# We will also need to load the class data and append it to the dataframe, for this we will use Pandas.

# In[ ]:


class_df = pd.read_csv('../input/stage_1_detailed_class_info.csv')

train_demo_df = pd.merge(left=train_demo_df, right=class_df, left_on='patientId', right_on='patientId')


# Let's quickly inspect the integer columns to see if there are any outliers.

# In[ ]:


print(train_demo_df.describe())
train_demo_df.head()


# Ok since only a handful of people make it past 110 years of age lets assume that the max patient age of 155 is an error, lets view all the images where the age is over 100

# In[ ]:


for file_name in tqdm(os.listdir('../input/stage_1_train_images/')):
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)
    if int(dataset.PatientAge) >= 100:
        show_dcm_info(dataset)
        plot_pixel_array(dataset)


# I was hoping these images would show smaller people and we could assume they were children and that the age was in months, it would a simple way to fix it. After doing a few comparisons to different age groups I will assume that the 1 has been placed there in error. They appear to be too old to by under 13 and they appear to young to be over 80. To fix this we will clean the data and replace any values over 100 with the last two digits, making the 155 example 55 years old. 

# In[ ]:


train_demo_df = train_demo_df.where(train_demo_df['patientAge'] <= 100, train_demo_df['patientAge']-100, axis=1)


# Ok now we've decided and implemented out data cleansing, lets graph the distributions for Age and Sex using a Seaborn pairplot!

# In[ ]:


sns.pairplot(train_demo_df, hue='class', height=3)


# From these charts we can see that there are more images for males than females, both genders have the most classes of "No Lung Opacity/Not Normal", however besides this fact the men are more likely to have a class of "Lung Opacity" where as women are by proportion less likely.
# 
# to explain what I mean a little clearer see the list below:
# 
#   * Ranking of class probability in females:
#       - No Lung Opacity/Not Normal
#       - Normal
#       - Lung Opacity
#   * Ranking of class probability in males:
#      - No Lung Opacity/Not Normal
#      - Lung Opacity
#      - Normal
# 
# Lung opacity seems to also be distributed with a slight twin peaks shape, from the graph you can see that people around the age of 30 are the slightly more likely to have Lung Opacity for the age group below 40.
# 
# Another interesting note from these charts, is that some of those statistic values we extracted fro the images show clear and distinctive seperations between the classes. It would appear as if having a higher mean, and a higher minimum that the images are more likely to be classed as either "Normal" or "No Lung Opacity/Not Normal".

# **TO BE CONTINUED**

# In[ ]:




