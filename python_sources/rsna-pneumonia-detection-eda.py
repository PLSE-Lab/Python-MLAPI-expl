#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">RSNA Pneumonia Detection EDA</font></center></h1>
# 
# <center><img src="https://www.rsna.org/images/rsna/home/line_r.svg" width="500"></img></center>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#     -<a href='#21'>Load packages</a>  
#      -<a href='#21'>Load the data</a>  
# - <a href='#3'>Data exploration</a>   
#     -<a href='#31'>Missing data</a>  
#     -<a href='#32'>Merge train and class info data</a>  
#     -<a href='#33'>Explore DICOM data</a>  
#     -<a href='#34'>Add meta information from DICOM data</a>  
#     -<a href='#35'>Modality</a>  
#     -<a href='#36'>Body Part Examined</a>  
#     -<a href='#37'>View Position</a>  
#     -<a href='#38'>Conversion Type</a>  
#     -<a href='#39'>Rows and Columns</a>  
#     -<a href='#310'>Patient Age</a>  
#     -<a href='#311'>Patient Sex</a>  
# - <a href='#4'>Conclusions</a>    
# - <a href='#5'>References</a>    
# 

# In[ ]:


from datetime import datetime
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"Updated {dt_string} (GMT)")


# # <a id="1">Introduction</a>  
# 
# This Kernel objective is to explore the dataset for RSNA Pneumonia Detection Challenge.   
# 
# We start by exploring the DICOM data, we extract then meta information from the DICOM files and visualize the various features of the DICOM images, grouped by age, sex.
# 
# The Kernel was modified to work with **stage_2** data instead of **stage_1** data.
# 
# 

# # <a id="2">Prepare the data analysis</a>  
# 
# ## <a id="21">Load packages</a>
# 

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from matplotlib.patches import Rectangle
import seaborn as sns
import pydicom as dcm
get_ipython().run_line_magic('matplotlib', 'inline')
IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/rsna-pneumonia-detection-challenge"
else:
    PATH="../input/"
print(os.listdir(PATH))


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="22">Load the data</a>
# 
# Let's load the tabular data. There are two files:
# * Detailed class info;  
# * Train labels.

# In[ ]:


class_info_df = pd.read_csv(PATH+'/stage_2_detailed_class_info.csv')
train_labels_df = pd.read_csv(PATH+'/stage_2_train_labels.csv')                         


# In[ ]:


print(f"Detailed class info -  rows: {class_info_df.shape[0]}, columns: {class_info_df.shape[1]}")
print(f"Train labels -  rows: {train_labels_df.shape[0]}, columns: {train_labels_df.shape[1]}")


# Let's explore the two loaded files. We will take out a 5 rows samples from each dataset.

# In[ ]:


class_info_df.sample(10)


# In[ ]:


train_labels_df.sample(10)


# In **class detailed info** dataset are given the detailed information about the type of positive or negative class associated with a certain patient.  
# 
# In **train labels** dataset are given the patient ID and the window (x min, y min, width and height of the) containing evidence of pneumonia.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="1">Data exploration</a>  
# 
# Let's explore the data further.

# ## <a id="31">Missing data</a>
# 
# Let's check missing information in the two datasets. 

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))
missing_data(train_labels_df)


# In[ ]:


missing_data(class_info_df)


# The percent missing for x,y, height and width in train labels represents the percent of the target **0** (not **Lung opacity**).
# 
# Let's check the class distribution from class detailed info.

# In[ ]:


f, ax = plt.subplots(1,1, figsize=(6,4))
total = float(len(class_info_df))
sns.countplot(class_info_df['class'],order = class_info_df['class'].value_counts().index, palette='Set3')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(100*height/total),
            ha="center") 
plt.show()


# Let's look into more details to the classes.

# In[ ]:


def get_feature_distribution(data, feature):
    # Get the count for each label
    label_counts = data[feature].value_counts()

    # Get total number of samples
    total_samples = len(data)

    # Count the number of items in each class
    print("Feature: {}".format(feature))
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000) / 100
        print("{:<30s}:   {} or {}%".format(label, count, percent))

get_feature_distribution(class_info_df, 'class')


# **No Lung Opacity / Not Normal** and **Normal** have together the same percent (**69.077%**) as the percent of missing values for target window in class details information.   
# 
# In the train set, the percent of data with value for **Target = 1** is therefore **30.92%**.   
# 

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="32">Merge train and class detail info data</a>   
# 
# Let's merge now the two datasets, using Patient ID as the merge criteria.

# In[ ]:


train_class_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')


# In[ ]:


train_class_df.sample(5)


# ### Target and class  
# 
# Let's plot the number of examinations for each class detected, grouped by Target value.

# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6))
tmp = train_class_df.groupby('Target')['class'].value_counts()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
sns.barplot(ax=ax,x = 'Target', y='Exams',hue='class',data=df, palette='Set3')
plt.title("Chest exams class and Target")
plt.show()


# All chest examinations with`Target` = **1** (pathology detected) associated with `class`:  **Lung Opacity**.    
# 
# The chest examinations with `Target` = **0** (no pathology detected) are either of `class`: **Normal** or `class`: **No Lung Opacity / Not Normal**.

# ### Detected Lung Opacity window   
# 
# For the class **Lung Opacity**, corresponding to values of **Target = 1**, we plot the density of **x**, **y**, **width** and **height**.
# 
# 

# In[ ]:


target1 = train_class_df[train_class_df['Target']==1]
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(12,12))
sns.distplot(target1['x'],kde=True,bins=50, color="red", ax=ax[0,0])
sns.distplot(target1['y'],kde=True,bins=50, color="blue", ax=ax[0,1])
sns.distplot(target1['width'],kde=True,bins=50, color="green", ax=ax[1,0])
sns.distplot(target1['height'],kde=True,bins=50, color="magenta", ax=ax[1,1])
locs, labels = plt.xticks()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()


# We can plot also the center of the rectangles points in the plane x0y.   The centers of the rectangles are the points $$x_c = x + \frac{width}{2}$$ and $$y_c = y + \frac{height}{2}$$.
# 
# We will show a sample of center points superposed with the corresponding sample of the rectangles.
# The rectangles are created using the method described in Kevin's Kernel <a href="#4">[1]</a>.

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(7,7))
target_sample = target1.sample(2000)
target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2
target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2
plt.title("Centers of Lung Opacity rectangles (brown) over rectangles (yellow)\nSample size: 2000")
target_sample.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color="brown")
for i, crt_sample in target_sample.iterrows():
    ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),
                width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color="yellow"))
plt.show()


# We follow with the exploration of the DICOM data.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="33">Explore DICOM data</a>  
# 
# Let's read now the DICOM data in the train set. The image path is as following:

# In[ ]:


image_sample_path = os.listdir(PATH+'/stage_2_train_images')[:5]
print(image_sample_path)


# The files names are the patients IDs.    
# Let's check how many images are in the train and test folders.

# In[ ]:


image_train_path = os.listdir(PATH+'/stage_2_train_images')
image_test_path = os.listdir(PATH+'/stage_2_test_images')
print("Number of images in train set:", len(image_train_path),"\nNumber of images in test set:", len(image_test_path))


# 
# 
# Only a reduced number of images are present in the training set (**26684**), compared with the number of  images in the train_df data (**30227**).  
# 
# It might be that we do have duplicated entries in the train and class datasets. Let's check this.
# 
# ### Check duplicates in train dataset
# 

# In[ ]:


print("Unique patientId in  train_class_df: ", train_class_df['patientId'].nunique())      


# We confirmed that the number of *unique* **patientsId** are equal with the number of DICOM images in the train set.  
# 
# Let's see what entries are duplicated. We want to check how are these distributed accross classes and Target value.

# In[ ]:


tmp = train_class_df.groupby(['patientId','Target', 'class'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target','class']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df2.columns = ['Exams', 'Target','Class', 'Entries']
df2


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6))
sns.barplot(ax=ax,x = 'Target', y='Entries', hue='Exams',data=df2, palette='Set2')
plt.title("Chest exams class and Target")
plt.show()


# 
# Let's now extract one image and process the DICOM information. 

# ### DICOM meta data

# In[ ]:


samplePatientID = list(train_class_df[:3].T.to_dict().values())[0]['patientId']
samplePatientID = samplePatientID+'.dcm'
dicom_file_path = os.path.join(PATH,"stage_2_train_images/",samplePatientID)
dicom_file_dataset = dcm.read_file(dicom_file_path)
dicom_file_dataset


# We can observe that we do have available some useful information in the DICOM metadata with predictive value, for example:   
# * Patient sex;   
# * Patient age;  
# * Modality;  
# * Body part examined;  
# * View position;  
# * Rows & Columns;  
# * Pixel Spacing.  
# 

# Let's sample few images having the **Target = 1**.
# 
# ### Plot DICOM images with Target = 1

# In[ ]:


def show_dicom_images(data):
    img_data = list(data.T.to_dict().values())
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(img_data):
        patientImage = data_row['patientId']+'.dcm'
        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}\nWindow: {}:{}:{}:{}'.format(
                data_row['patientId'],
                modality, age, sex, data_row['Target'], data_row['class'], 
                data_row['x'],data_row['y'],data_row['width'],data_row['height']))
    plt.show()


# In[ ]:


show_dicom_images(train_class_df[train_class_df['Target']==1].sample(9))


# We would like to represent the images with the overlay boxes superposed. For this, we will need first to parse the whole dataset with **Target = 1** and gather all coordinates of the windows showing a **Lung Opacity** on the same image.  The simples method is show in <a href='#5'>[1]</a> and we will adapt our rendering from this method.

# In[ ]:


def show_dicom_images_with_boxes(data):
    img_data = list(data.T.to_dict().values())
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(img_data):
        patientImage = data_row['patientId']+'.dcm'
        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(
                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))
        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]
        box_data = list(rows.T.to_dict().values())
        for j, row in enumerate(box_data):
            ax[i//3, i%3].add_patch(Rectangle(xy=(row['x'], row['y']),
                        width=row['width'],height=row['height'], 
                        color="yellow",alpha = 0.1))   
    plt.show()


# In[ ]:


show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1].sample(9))


# For some of the images with **Target=1**, we might see multiple areas (boxes/rectangles) with **Lung Opacity**.
# 
# Let's sample few images having the **Target = 0**.   
# 
# ### Plot DICOM images with Target = 0
# 

# In[ ]:


show_dicom_images(train_class_df[train_class_df['Target']==0].sample(9))


# <a href="#0"><font size="1">Go to top</font></a>   
# 
# 
# ## <a id="34">Add meta information from DICOM data</a>
# 
# 
# ### Train data
# 
# We will parse the DICOM meta information and add it to the train dataset. We will do the same with the test data.

# In[ ]:


vars = ['Modality', 'PatientAge', 'PatientSex', 'BodyPartExamined', 'ViewPosition', 'ConversionType', 'Rows', 'Columns', 'PixelSpacing']

def process_dicom_data(data_df, data_path):
    for var in vars:
        data_df[var] = None
    image_names = os.listdir(PATH+data_path)
    for i, img_name in tqdm_notebook(enumerate(image_names)):
        imagePath = os.path.join(PATH,data_path,img_name)
        data_row_img_data = dcm.read_file(imagePath)
        idx = (data_df['patientId']==data_row_img_data.PatientID)
        data_df.loc[idx,'Modality'] = data_row_img_data.Modality
        data_df.loc[idx,'PatientAge'] = pd.to_numeric(data_row_img_data.PatientAge)
        data_df.loc[idx,'PatientSex'] = data_row_img_data.PatientSex
        data_df.loc[idx,'BodyPartExamined'] = data_row_img_data.BodyPartExamined
        data_df.loc[idx,'ViewPosition'] = data_row_img_data.ViewPosition
        data_df.loc[idx,'ConversionType'] = data_row_img_data.ConversionType
        data_df.loc[idx,'Rows'] = data_row_img_data.Rows
        data_df.loc[idx,'Columns'] = data_row_img_data.Columns  
        data_df.loc[idx,'PixelSpacing'] = str.format("{:4.3f}",data_row_img_data.PixelSpacing[0]) 


# In[ ]:


process_dicom_data(train_class_df,'stage_2_train_images/')


# ### Test data
# 
# We will create as well a test dataset with similar information.

# In[ ]:


test_class_df = pd.read_csv(PATH+'/stage_2_sample_submission.csv')


# In[ ]:


test_class_df = test_class_df.drop('PredictionString',1)
process_dicom_data(test_class_df,'stage_2_test_images/')


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="35">Modality</a>
# 
# Let's check how many modalities are used. Both train and test set are checked.

# In[ ]:


print("Modalities: train:",train_class_df['Modality'].unique(), "test:", test_class_df['Modality'].unique())


# The meaning of this modality is **CR** - **Computer Radiography**  <a href='#4'>[2]</a> <a href='#4'>[3]</a>.
# 
# 
# ## <a id="36">Body Part Examined</a>
# 
# Let's check if other body parts than 'CHEST' appears in the data.

# In[ ]:


print("Body Part Examined: train:",train_class_df['BodyPartExamined'].unique(), "test:", test_class_df['BodyPartExamined'].unique())


# ## <a id="37">View Position</a>
# 
# View Position is a radiographic view associated with the Patient Position. Let's check the View Positions distribution for the both datasets.
# 
# 
# 

# In[ ]:


print("View Position: train:",train_class_df['ViewPosition'].unique(), "test:", test_class_df['ViewPosition'].unique())


# ### Train dataset  
# 
# Let's get into more details for the train dataset. First, let's check the distribution of PA and AP.

# In[ ]:


get_feature_distribution(train_class_df,'ViewPosition')


# Both **AP** and **PA** body positions are present in the data.  The meaning of these view positions are <a href='#4'>[2]</a> <a href='#4'>[3]</a>:
# * **AP** - Anterior/Posterior;    
# * **PA** - Posterior/Anterior.    
# 
# 
# Let's check, for the training data presenting **Lung Opacity**, the distribution of the window for both View Positions. We create a function to represent the distribution of the window centers and windows.

# In[ ]:


def plot_window(data,color_point, color_window,text):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    plt.title("Centers of Lung Opacity rectangles over rectangles\n{}".format(text))
    data.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color=color_point)
    for i, crt_sample in data.iterrows():
        ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),
            width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color=color_window))
    plt.show()


# We sample a subset of the train data with **Target = 1**. We calculate as well the center of the windows with **Lung Opacity**.   We then select from this sample the data with the two view position, to plot the window distribution separatelly.

# In[ ]:


target1 = train_class_df[train_class_df['Target']==1]

target_sample = target1.sample(2000)
target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2
target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2

target_ap = target_sample[target_sample['ViewPosition']=='AP']
target_pa = target_sample[target_sample['ViewPosition']=='PA']


# In[ ]:


plot_window(target_ap,'green', 'yellow', 'Patient View Position: AP')


# In[ ]:


plot_window(target_pa,'blue', 'red', 'Patient View Position: PA')


# ### Test dataset  
# 
# Let's check the distribution of AP and PA positions for the test set.

# In[ ]:


get_feature_distribution(test_class_df,'ViewPosition')


# ## <a id="38">Conversion Type</a>
# 
# Let's check the Conversion Type data.

# In[ ]:


print("Conversion Type: train:",train_class_df['ConversionType'].unique(), "test:", test_class_df['ConversionType'].unique())


# Both train and test have only **WSD** Conversion Type Data. The meaning of this Conversion Type is **WSD**: **Workstation**.
# 
# ## <a id="39">Rows and Columns</a>

# In[ ]:


print("Rows: train:",train_class_df['Rows'].unique(), "test:", test_class_df['Rows'].unique())
print("Columns: train:",train_class_df['Columns'].unique(), "test:", test_class_df['Columns'].unique())


# Only {Rows:Columns} {1024:1024} are present in both train and test.  
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>

# ## <a id="310">Patient Age</a>
# 
# Let's examine now the data for the Patient Age for the train set.
# 
# ### Train dataset

# In[ ]:


tmp = train_class_df.groupby(['Target', 'PatientAge'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target', 'PatientAge']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()


# In[ ]:


tmp = train_class_df.groupby(['class', 'PatientAge'])['patientId'].count()
df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df1.groupby(['Exams','class', 'PatientAge']).count()
df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()


# In[ ]:


fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='Target',data=df2)
plt.title("Train set: Chest exams Age and Target")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='class',data=df3)
plt.title("Train set: Chest exams Age and class")
plt.xticks(rotation=90)
plt.show()


# 
# **Note**: most probably, the values of age 148 to 155 are mistakes.   
# 
# Let's group the ages in 5 groups (0-19, 20-34, 35-49, 50-64 and 65+). 

# In[ ]:


target_age1 = target_sample[target_sample['PatientAge'] < 20]
target_age2 = target_sample[(target_sample['PatientAge'] >=20) & (target_sample['PatientAge'] < 35)]
target_age3 = target_sample[(target_sample['PatientAge'] >=35) & (target_sample['PatientAge'] < 50)]
target_age4 = target_sample[(target_sample['PatientAge'] >=50) & (target_sample['PatientAge'] < 65)]
target_age5 = target_sample[target_sample['PatientAge'] >= 65]


# Let's show the distribution of windows for the 5 age groups.

# In[ ]:


plot_window(target_age1,'blue', 'red', 'Patient Age: 1-19 years')


# In[ ]:


plot_window(target_age2,'blue', 'red', 'Patient Age: 20-34 years')


# In[ ]:


plot_window(target_age3,'blue', 'red', 'Patient Age: 35-49 years')


# In[ ]:


plot_window(target_age4,'blue', 'red', 'Patient Age: 50-65 years')


# In[ ]:


plot_window(target_age5,'blue', 'red', 'Patient Age: 65+ years')


# Let's check also the distribution of patient age for the test data set.
# 
# ### Test dataset

# In[ ]:


fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.countplot(test_class_df['PatientAge'], ax=ax)
plt.title("Test set: Patient Age")
plt.xticks(rotation=90)
plt.show()


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="311">Patient Sex</a>
# 
# Let's examine now the data for the Patient Sex.   
# 
# ### Train dataset
# 
# We represent the number of Exams for each Patient Sex, grouped by value of Target.

# In[ ]:


tmp = train_class_df.groupby(['Target', 'PatientSex'])['patientId'].count()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['Exams','Target', 'PatientSex']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, ax = plt.subplots(nrows=1,figsize=(6,6))
sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='Target',data=df2)
plt.title("Train set: Patient Sex and Target")
plt.show()


# We represent the number of Exams for each Patient Sex, grouped by value of  class.

# In[ ]:


tmp = train_class_df.groupby(['class', 'PatientSex'])['patientId'].count()
df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
tmp = df1.groupby(['Exams','class', 'PatientSex']).count()
df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, (ax) = plt.subplots(nrows=1,figsize=(6,6))
sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='class',data=df3)
plt.title("Train set: Patient Sex and class")
plt.show()


# Let's plot as well the distribution of  window with Lung Opacity, separatelly for the female and male patients. We will reuse the sample with **Target = 1** for which we calculated also the center of the window.

# In[ ]:


target_female = target_sample[target_sample['PatientSex']=='F']
target_male = target_sample[target_sample['PatientSex']=='M']


# In[ ]:


plot_window(target_female,"red", "magenta","Patients Sex: Female")


# In[ ]:


plot_window(target_male,"darkblue", "blue", "Patients Sex: Male")


# Let's check as well the distribution of Patient Sex for the test data.   
# 
# ### Test dataset

# In[ ]:


sns.countplot(test_class_df['PatientSex'])
plt.title("Test set: Patient Sex")
plt.show()


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# # <a id='4'>Conclusions</a>   
# 
# After exploring the data, both the tabular and DICOM data, we were able to:  
# - discover duplications in the tabular data;  
# - explore the DICOM images;  
# - extract meta information from the DICOM data;  
# - add features to the tabular data from the meta information in DICOM data;  
# - further analyze the distribution of the data with the newly added features from DICOM metadata;  
# 
# All these findings are useful as preliminary work for building a model.
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id='5'>References</a>  
# 
# 
# [1] Kevin Mader, Lung Opacity Overview, https://www.kaggle.com/kmader/lung-opacity-overview  
# [2] Modality Specific Modules, DICOM Standard,  http://dicom.nema.org/medical/dicom/2014c/output/chtml/part03/sect_C.8.html  
# [3] DICOM Standard, https://www.dicomstandard.org/     
# [4] Getting Started with Pydicom, https://pydicom.github.io/pydicom/stable/getting_started.html   
# [5] ITKPYthon package, https://itkpythonpackage.readthedocs.io/en/latest/   
# [6] DICOM in Python: Importing medical image data into NumPy with PyDICOM and VTK, https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/  
# [7] DICOM Processing and Segmentation in Python, https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/  
# [8] DICOM Standard Browser, https://dicom.innolitics.com/ciods  
# [9] How can I read a DICOM image in Python, https://www.quora.com/How-can-I-read-a-DICOM-image-in-Python  
# [10] DICOM read example in Python, https://www.programcreek.com/python/example/97517/dicom.read_file    
# [11] DICOM in Python, https://github.com/pydicom   
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>
# 
