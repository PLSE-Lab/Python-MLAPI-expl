#!/usr/bin/env python
# coding: utf-8

# # Ohoh! Don't turn into a smoothie! 
# 
# ## The shake-up is likely to come! :-)
# 
# Take a look at some EDA findings to observe differences in train and test data. Be careful to not overfit too badly to the training data and keep in mind the train/test differences. 

# 
# <img src="https://cdn.pixabay.com/photo/2017/04/23/09/44/smoothies-2253423_1280.jpg" width="900px">
# 
# 

# ## Table of contents
# 
# 1. [Important findings](#findings)
# 2. [Prepare to start](#prepare) 
# 3. [What is given by the meta data?](#meta_data) 
#     * [Missing values](#missing_vals) 
#     * [Image names](#images_names) 
#     * [Patient id counts](#patient_ids)
#     * [Overlapping patients in train/test](#overlapping_patients)
#     * [Gender counts](#gender_counts)
#     * [Age distributions](#age_distributions)
#     * [Image location](#image_location)
# 4. [Feature-feature interactions](#interactions)
#     * [Ages per patient](#ages_per_patient)
#     * [Age and gender](#age_gender)
#     * [Age, gender and cancer](#age_gender_cancer)
#     * [Individual patient information](#patient_information)
# 5. [How do train and test set images differ?](#train_test_images_eda)
#     * [File structure and dicom images](#file_structure)
#     * [Train and test image EDA](#images_eda) 
# 6. [Building up the model](#modelling)
#     * [Validation strategy](#validation)
#     * [Dataset](#datasetloader)
#     * [Augmentations](#augmentations)
#     * [Loss and evaluation](#loss)
#     * [Model structure](#model)
#     * [Predict on whatever you like](#predict)
#     * [Training loop](#training_loop)
#     * [Searching for an optimal learning rate](#learning_rate_search)
# 7. [Experimental zone](#experimental_zone)
#     * [Creating a hold-out dataset](#hold_out)
#     * [Settings](#settings)
#     * [Using Bojans resized images to speed up](#resized_images)
#     * [Searching for learning rate boundaries](#lr_bounds)
#     * [Running a model](#running)
#     * [Exploring predictions and weaknesses](#result_analysis)
#     * [Submission](#submission)
# 8. [Final conclusion](#conclusion)
#     

# # Important findings <a class="anchor" id="findings"></a>
# 
# * **We can clearly observe groups of images with similar statistics that depend on the image shapes!!!**
# * **There is one group of the test images that is missing in train(1080 rows 1920 columns)! This would mean a complete new type of images that may lead to notable differences in CV scores and LB!**
# * For most of the patients there were only a few images recorded (range from 1 to 20).
# * 5% of the patients show more than 45 images. There is an extreme outlier in the test data with roughly 250 images!
# * We have more males than females in both train and test data. For the test set this imbalance is even higher!
# * We can observe **more older patients in test than in train**! The age is normally distributed in train but shows multiple modes in test.
# * The distributions of image locations look very similar for train and test.
# * We have highly imbalanced target classes!
# * Multiple images does not mean that there are multiple ages involved! 
# * We can observe a high surplus of males in the ages 45 to 50 and 70, 75 in train and test but in test we can find **even more males of high age > 75**.
# * We have more malignant cases of higher age than benign cases.
# * 62 % of the malignant cases belong to males and only 38 % to females! **We have to be very careful!!! As we have a surpus of males with ages above 70 and 75 it's unclear if the sex is really an important feature for having melanoma or not.** It could also be that the age is most important and that we only have more malignant cases for males due to their higher age! 

# # Prepare to start <a class="anchor" id="prepare"></a>

# In[ ]:


get_ipython().system('pip install efficientnet_pytorch torchtoolbox')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import seaborn as sns
sns.set()

import re

import pydicom
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet

from scipy.special import softmax

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, auc

from skimage.io import imread
from PIL import Image

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import os
import copy

from albumentations import Compose, RandomCrop, Normalize,HorizontalFlip, Resize
from albumentations import VerticalFlip, RGBShift, RandomBrightness
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensor

from tqdm.notebook import tqdm

os.listdir("../input/")


# In[ ]:


basepath = "../input/siim-isic-melanoma-classification/"
modelspath = "../input/pytorch-pretrained-image-models/"
imagestatspath = "../input/siimisic-melanoma-classification-image-stats/"


# In[ ]:


os.listdir(basepath)


# In[ ]:


train_info = pd.read_csv(basepath + "train.csv")
train_info.head()


# In[ ]:


test_info = pd.read_csv(basepath + "test.csv")
test_info.head()


# Our test set misses three columns: diagnosis, benign_malignant & target.

# In[ ]:


train_info.shape[0] / test_info.shape[0]


# Three times more entries in train than in test.

# # What is given by the meta data? <a class="anchor" id="meta_data"></a>

# ## Missing values <a class="anchor" id="missing_vals"></a>

# In[ ]:


missing_vals_train = train_info.isnull().sum() / train_info.shape[0]
missing_vals_train[missing_vals_train > 0].sort_values(ascending=False)


# In[ ]:


missing_vals_test = test_info.isnull().sum() / test_info.shape[0]
missing_vals_test[missing_vals_test > 0].sort_values(ascending=False)


# The anatomy shows most missing values. 

# ## Image names <a class="anchor" id="image_names"></a>

# In[ ]:


train_info.image_name.value_counts().max()


# In[ ]:


test_info.image_name.value_counts().max()


# Ok, great, all names are unique.

# ## Patient id counts <a class="anchor" id="patient_ids"></a>

# In[ ]:


train_info.patient_id.value_counts().max()


# In[ ]:


test_info.patient_id.value_counts().max()


# In contrast we can find multiple images for one patient!

# In[ ]:


patient_counts_train = train_info.patient_id.value_counts()
patient_counts_test = test_info.patient_id.value_counts()

fig, ax = plt.subplots(2,2,figsize=(20,12))

sns.distplot(patient_counts_train, ax=ax[0,0], color="orangered", kde=True);
ax[0,0].set_xlabel("Counts")
ax[0,0].set_ylabel("Frequency")
ax[0,0].set_title("Patient id value counts in train");

sns.distplot(patient_counts_test, ax=ax[0,1], color="lightseagreen", kde=True);
ax[0,1].set_xlabel("Counts")
ax[0,1].set_ylabel("Frequency")
ax[0,1].set_title("Patient id value counts in test");

sns.boxplot(patient_counts_train, ax=ax[1,0], color="orangered");
ax[1,0].set_xlim(0, 250)
sns.boxplot(patient_counts_test, ax=ax[1,1], color="lightseagreen");
ax[1,1].set_xlim(0, 250);


# In[ ]:


np.quantile(patient_counts_train, 0.75) - np.quantile(patient_counts_train, 0.25)


# In[ ]:


np.quantile(patient_counts_train, 0.5)


# In[ ]:


print(np.quantile(patient_counts_train, 0.95))
print(np.quantile(patient_counts_test, 0.95))


# ### Insights
# 
# * For most of the patients we only have a few images ranging fom 1 to roughly 20.
# * More than 45 images per patient is very seldom! 
# * Nonetheless we have patients with more than 100 images.
# * There is one heavy outlier patient in the test set with close to 250 images.

# In[ ]:


200/test_info.shape[0] * 100


# This outlier patient holds ~1.8 % of the test data! Ohoh! :-O

# ## Overlapping patients in train/test <a class="anchor" id="overlapping_patients"></a>

# In[ ]:


train_patient_ids = set(train_info.patient_id.unique())
test_patient_ids = set(test_info.patient_id.unique())

train_patient_ids.intersection(test_patient_ids)


# Ok, that's great! There seem to be no patients in train that can be found in test as well. We can't be sure as we don't know how the naming assignment process was designed. We might better check the images themselves as well!

# ## Gender counts  <a class="anchor" id="gender_counts"></a>

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train_info.sex, palette="Reds_r", ax=ax[0]);
ax[0].set_xlabel("")
ax[0].set_title("Gender counts");

sns.countplot(test_info.sex, palette="Blues_r", ax=ax[1]);
ax[1].set_xlabel("")
ax[1].set_title("Gender counts");


# ### Insights
# 
# * We observe more males than females in both train and test data.
# * The surplus of males is even higher in test than in train!

# ## Age distributions <a class="anchor" id="age_distributions"></a>

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_info.age_approx, color="orangered", ax=ax[0]);
labels = ax[0].get_xticklabels();
ax[0].set_xticklabels(labels, rotation=90);
ax[0].set_xlabel("");
ax[0].set_title("Age distribution in train");

sns.countplot(test_info.age_approx, color="lightseagreen", ax=ax[1]);
labels = ax[1].get_xticklabels();
ax[1].set_xticklabels(labels, rotation=90);
ax[1].set_xlabel("");
ax[1].set_title("Age distribution in test");


# ### Insights
# 
# * The age distribution in train looks almost normally distributed.
# * In contrast, the age distribution in test shows multiple modes and interesting peaks at the ageof 55 and 70!
# * We can observe more older patients in test than in train! This kind of imbalance can be important for our model performance if the age is an important feature.  

# ## Image location <a class="anchor" id="image_location"></a>

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))

image_locations_train = train_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)
image_locations_test = test_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

sns.barplot(x=image_locations_train.index.values, y=image_locations_train.values, ax=ax[0], color="orangered");
ax[0].set_xlabel("");
labels = ax[0].get_xticklabels();
ax[0].set_xticklabels(labels, rotation=90);
ax[0].set_title("Image locations in train");

sns.barplot(x=image_locations_test.index.values, y=image_locations_test.values, ax=ax[1], color="lightseagreen");
ax[1].set_xlabel("");
labels = ax[1].get_xticklabels();
ax[1].set_xticklabels(labels, rotation=90);
ax[1].set_title("Image locations in test");


# ### Insights
# 
# * The distributions of image locations in train and test look very similar. 
# * Most images are related to the torso or to the lower extremity.

# ## Target distribution <a class="anchor" id="target_distribution"></a>

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,5))

sns.countplot(x=train_info.diagnosis, orient="v", ax=ax[0], color="Orangered")
ax[0].set_xlabel("")
labels = ax[0].get_xticklabels();
ax[0].set_xticklabels(labels, rotation=90);
ax[0].set_title("Diagnosis");

sns.countplot(train_info.benign_malignant, ax=ax[1], palette="Reds_r");
ax[1].set_xlabel("")
ax[1].set_title("Type");


# ### Insights
# 
# * The diagnosis is often unknown and for those known we observe a very high imbalance. Most likely we can't expect much from this additional target feature.
# * The target is highly imbalanced and we have to find proper strategies to deal with this kind of target distribution during learning.

# In[ ]:


train_info.groupby("benign_malignant").target.nunique()


# The benign_malignant column is the same as the target.

# # Feature-feature interactions <a class="anchor" id="interactions"></a>

# ## Ages per patient <a class="anchor" id="ages_per_patient"></a>

# In[ ]:


patient_ages_table_train = train_info.groupby(["patient_id", "age_approx"]).size() / train_info.groupby("patient_id").size()
patient_ages_table_train = patient_ages_table_train.unstack().transpose()
patient_ages_table_test = test_info.groupby(["patient_id", "age_approx"]).size() / test_info.groupby("patient_id").size()
patient_ages_table_test = patient_ages_table_test.unstack().transpose()

patient_with_known_ages_train = train_info[train_info.patient_id.isin(patient_ages_table_train.columns.values)]

sorted_patients_train = patient_with_known_ages_train.patient_id.value_counts().index.values
patient_with_known_ages_test = test_info[test_info.patient_id.isin(patient_ages_table_test.columns.values)]
sorted_patients_test = patient_with_known_ages_test.patient_id.value_counts().index.values

fig, ax = plt.subplots(2,1, figsize=(20,20))
sns.heatmap(patient_ages_table_train[sorted_patients_train], cmap="Reds", ax=ax[0], cbar=False);
ax[0].set_title("Image coverage in % per patient and age in train data");
sns.heatmap(patient_ages_table_test[sorted_patients_test], cmap="Blues", ax=ax[1], cbar=False);
ax[1].set_title("Image coverage in % per patient and age in test data");
ax[0].set_xlabel("")
ax[1].set_xlabel("");


# ### Insights
# 
# * Be careful with interpreting these heatmaps: 
#     * **The patients are soreted by value_counts**. Patients with the most number of images are given on the left and those with only a few or a single images are on the righthand side.
#     * The color represents how much percentage of the images for one patient is covered by a given age. 
#     * For example the most left patient in test is the one with almost 250 images. The related images are not spread over a wide range of different ages and are very concentrated at an old age. (Dark blue color at the age of 70).    
# * We can conclude that more images does not mean that there are multiple ages involved! 
# * It's possible that multiple images are spread over a wide range of ages but it's also possible that multiple images are concentrated at one age.

# ## Age and gender <a class="anchor" id="age_gender"></a>

# Looking at the sex per patient (excluding the multiple counts due to multiple images) we can observe that we still have more males than females.

# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,15))

sns.boxplot(train_info.sex, train_info.age_approx, ax=ax[0,0], palette="Reds_r");
ax[0,0].set_title("Age per gender in train");

sns.boxplot(test_info.sex, test_info.age_approx, ax=ax[0,1], palette="Blues_r");
ax[0,1].set_title("Age per gender in test");

sns.countplot(train_info.age_approx, hue=train_info.sex, ax=ax[1,0], palette="Reds_r");
sns.countplot(test_info.age_approx, hue=test_info.sex, ax=ax[1,1], palette="Blues_r");


# ### Insights
# There are some significant differences in train and test regarding the gender per age level:
# 
# * At the ages between 25 and 35 we have much more females than males in train but a balanced count in test!
# * We can observe a high surplus of males in the ages 45 to 50 and 70, 75 in train and test but in test we can find even more males of high age > 75.

# ## Age, gender and cancer <a class="anchor" id="age_gender_cancer"></a>

# In[ ]:


sex_and_cancer_map = train_info.groupby(
    ["benign_malignant", "sex"]
).size().unstack(level=0) / train_info.groupby("benign_malignant").size() * 100

cancer_sex_map = train_info.groupby(
    ["benign_malignant", "sex"]
).size().unstack(level=1) / train_info.groupby("sex").size() * 100


fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.boxplot(train_info.benign_malignant, train_info.age_approx, ax=ax[0], palette="Greens");
ax[0].set_title("Age and cancer");
ax[0].set_xlabel("");

sns.heatmap(sex_and_cancer_map, annot=True, cmap="Greens", cbar=False, ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_ylabel("");

sns.heatmap(cancer_sex_map, annot=True, cmap="Greens", cbar=False, ax=ax[2])
ax[2].set_xlabel("")
ax[2].set_ylabel("");


# ### Insights
# 
# * We have more malignant cases of higher age than benign cases.
# * 62 % of the malignant cases belong to males and only 38 % to females.
# * Roughly 2 % of the males in the train dataset show malignant cases, but only 1.4 % of the females.
# 
# We have to be very careful!!! As we have a surpus of males with ages above 70 and 75 it's unclear if the sex is really an important feature for having melanoma or not. It could also be that the age is most important and that we only have more malignant cases for males due to their higher age!  

# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,15))

sns.countplot(train_info[train_info.benign_malignant=="benign"].age_approx, hue=train_info.sex, palette="Purples_r", ax=ax[0,0])
ax[0,0].set_title("Benign cases in train");

sns.countplot(train_info[train_info.benign_malignant=="malignant"].age_approx, hue=train_info.sex, palette="Oranges_r", ax=ax[0,1])
ax[0,1].set_title("Malignant cases in train");

sns.violinplot(train_info.sex, train_info.age_approx, hue=train_info.benign_malignant, split=True, ax=ax[1,0], palette="Greens_r");
sns.violinplot(train_info.benign_malignant, train_info.age_approx, hue=train_info.sex, split=True, ax=ax[1,1], palette="RdPu");


# ### Insights
# 
# * For the benign cases we can see that there is still a surplus of males in the ages of 45 and 70, but the other ones look quite good and balanced.
# * **In contrast we can find a high gender imbalance for a wide range of ages for the malignant cases!** That's really interesting and the features age and gender as well as their interaction with cancer are definitely some to play with during modelling.

# ## Individual patient information <a class="anchor" id="patient_information"></a>
# 
# Let's collect some information for each patient:
# 
# * the number of recorded images
# * the gender
# * the min, max age and the age span
# * the number of benign & malignant cases
# * the minimum and maximum age of a patient with malignant cases

# In[ ]:


patient_gender_train = train_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])
patient_gender_test = test_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])

train_patients = pd.DataFrame(index=patient_gender_train.index.values, data=patient_gender_train.values, columns=["sex"])
test_patients = pd.DataFrame(index=patient_gender_test.index.values, data=patient_gender_test.values, columns=["sex"])

train_patients.loc[:, "num_images"] = train_info.groupby("patient_id").size()
test_patients.loc[:, "num_images"] = test_info.groupby("patient_id").size()

train_patients.loc[:, "min_age"] = train_info.groupby("patient_id").age_approx.min()
train_patients.loc[:, "max_age"] = train_info.groupby("patient_id").age_approx.max()
test_patients.loc[:, "min_age"] = test_info.groupby("patient_id").age_approx.min()
test_patients.loc[:, "max_age"] = test_info.groupby("patient_id").age_approx.max()

train_patients.loc[:, "age_span"] = train_patients["max_age"] - train_patients["min_age"]
test_patients.loc[:, "age_span"] = test_patients["max_age"] - test_patients["min_age"]

train_patients.loc[:, "benign_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "benign"]
train_patients.loc[:, "malignant_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "malignant"]
train_patients["min_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.min().loc[:, "malignant"]
train_patients["max_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.max().loc[:, "malignant"]


# In[ ]:


train_patients.sort_values(by="malignant_cases", ascending=False).head()


# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,12))
sns.countplot(train_patients.sex, ax=ax[0,0], palette="Reds")
ax[0,0].set_title("Gender counts with unique patient ids in train")
sns.countplot(test_patients.sex, ax=ax[0,1], palette="Blues");
ax[0,1].set_title("Gender counts with unique patient ids in test");

train_age_span_perc = train_patients.age_span.value_counts() / train_patients.shape[0] * 100
test_age_span_perc = test_patients.age_span.value_counts() / test_patients.shape[0] * 100

sns.barplot(train_age_span_perc.index, train_age_span_perc.values, ax=ax[1,0], color="Orangered");
sns.barplot(test_age_span_perc.index, test_age_span_perc.values, ax=ax[1,1], color="Lightseagreen");
ax[1,0].set_title("Patients age span in train")
ax[1,1].set_title("Patients age span in test")
for n in range(2):
    ax[1,n].set_ylabel("% in data")
    ax[1,n].set_xlabel("age span");


# ### Insights
# 
# * Even on the patient id level we have more males than females in both train and test data.
# * The age span has more cases of 5 years in train than in test and less example with no age differences at all (age span of 0).

# # How do train and test set images differ? <a class="anchor" id="train_test_images_eda"></a> 

# ## File structure and dicom images <a class="anchor" id="file_structure"></a> 
# Let's take a look at the file structure:

# In[ ]:


example_files = os.listdir(basepath + "train/")[0:2]
example_files


# Ok, dicom images and the image names can be found in the train and test info (meta data) as well:

# In[ ]:


train_info.head(2)


# Perhaps we can do both: exploring the images and building up datasets and dataloaders for modelling. Let's start with the dataset and the corresponding dataframes. All we need it the imagepath and for the training data the target:

# In[ ]:


train_info["dcm_path"] = basepath + "train/" + train_info.image_name + ".dcm"
test_info["dcm_path"] = basepath + "test/" + test_info.image_name + ".dcm"


# In[ ]:


print(train_info.dcm_path[0])
print(test_info.dcm_path[0])


# Let's load an example:

# In[ ]:


example_dcm = pydicom.dcmread(train_info.dcm_path[2])
example_dcm


# Probably the most interesting information is given by Rows, Columns and Pixel Data.

# In[ ]:


image = example_dcm.pixel_array
print(image.shape)


# As reading the dicom image file is really slow, let's use the jpeg files:

# In[ ]:


train_info["image_path"] = basepath + "jpeg/train/" + train_info.image_name + ".jpg"
test_info["image_path"] = basepath + "jpeg/test/" + test_info.image_name + ".jpg"


# ## Train and test image EDA <a class="anchor" id="images_eda"></a> 
# 
# **Caution** Everything after this part is under construction! ;-)

# I have created a dataset that covers some simple image statistics for train and test set images. It's not complete at the moment and I will update it soon, but we can already do some EDA using it.

# In[ ]:


os.listdir(imagestatspath)


# In[ ]:


test_image_stats = pd.read_csv(imagestatspath +  "test_image_stats.csv")
test_image_stats.head(1)


# In[ ]:


train_image_stats_1 = pd.read_csv(imagestatspath + "train_image_stats_10000.csv")
train_image_stats_2 = pd.read_csv(imagestatspath + "train_image_stats_20000.csv")
train_image_stats_3 = pd.read_csv(imagestatspath + "train_image_stats_toend.csv")
train_image_stats_4 = train_image_stats_1.append(train_image_stats_2)
train_image_stats = train_image_stats_4.append(train_image_stats_3)
train_image_stats.shape


# ### Test image statistics
# 
# To get started I have taken the mean, std and skewness of each test image and performed a 3D-scatterplot. To understand wheather the result also depends on the image shape, I have colored the points with the value of columns each image has and added a text description to each point that holds the row value.

# In[ ]:


plot_test = True


# If you set plot_test to False the following scatter plot will show statistics of all training examples instead:

# In[ ]:


if plot_test:
    N = test_image_stats.shape[0]
    selected_data = test_image_stats
    my_title = "Test image statistics"
else:
    N = train_image_stats.shape[0]
    selected_data = train_image_stats
    my_title = "Train image statistics"

trace1 = go.Scatter3d(
    x=selected_data.img_mean.values[0:N], 
    y=selected_data.img_std.values[0:N],
    z=selected_data.img_skew.values[0:N],
    mode='markers',
    text=selected_data["rows"].values[0:N],
    marker=dict(
        color=selected_data["columns"].values[0:N],
        colorscale = "Jet",
        colorbar=dict(thickness=10, title="image columns", len=0.8),
        opacity=0.4,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = my_title,
    scene = dict(
        xaxis = dict(title="Image mean"),
        yaxis = dict(title="Image standard deviation"),
        zaxis = dict(title="Image skewness"),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# ### Insights
# 
# Uhh, that's a bit surprising... 
# 
# * One example - there are a lot of images with 6000 columns (red cluster) that show a high image mean close to 160, a wide range of standard deviations but a small range of skewness compared to other images with different column size.
# * There is also a very interesting kind of outlier image shape with 4288 columns that show extreme negative image skewnesses but have narrow image means and stds. 
# * Looking at the training examples it seems that one group of the test images is missing (1080 rows 1920 columns)! This would mean a complete new type of images that may lead to notable differences in CV scores and LB!
# 
# 
# To conclude - Performing a bit more EDA on image shapes and their relationships with image statistics and differences in train and test sets may be of great help to understand this dataset better and to choose proper modelling structures. 

# In[ ]:


test_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / test_image_stats.shape[0]


# In[ ]:


train_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / train_image_stats.shape[0]


# There is indeed a big group of test images (1716 images with 1080 rows 1920 columns) that is not presented in train (at least not with similar % of the data)!

# In[ ]:


examples1 = {"rows": 1080, "columns": 1920}
examples2 = {"rows": 4000, "columns": 6000}


# In[ ]:


selection1 = np.random.choice(test_image_stats[
    (test_image_stats["rows"]==examples1["rows"]) & (test_image_stats["columns"]==examples1["columns"])
].path.values, size=8, replace=False)

fig, ax = plt.subplots(2,4,figsize=(20,8))

for n in range(2):
    for m in range(4):
        path = selection1[m + n*4]
        dcm_file = pydicom.dcmread(path)
        image = dcm_file.pixel_array
        ax[n,m].imshow(image)
        ax[n,m].grid(False)


# This is the myterious unknown group of test images that holds 15% of the test data! Keep them in mind. ;-)

# In[ ]:


selection2 = np.random.choice(test_image_stats[
    (test_image_stats["rows"]==examples2["rows"]) & (test_image_stats["columns"]==examples2["columns"])
].path.values, size=8, replace=False)

fig, ax = plt.subplots(2,4,figsize=(20,6))

for n in range(2):
    for m in range(4):
        path = selection2[m + n*4]
        dcm_file = pydicom.dcmread(path)
        image = dcm_file.pixel_array
        ax[n,m].imshow(image)
        ax[n,m].grid(False)


# Ohoh! :-O It's really astonishing how well these images can be grouped given the image shapes! Browsing through the shapes above you can cleary observe these kind of groups. 

# # Building up the model <a class="anchor" id="modelling"></a>

# ## Dataset <a class="anchor" id="dataset"></a> 

# In[ ]:


class MelanomaDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]
        image = Image.open(path)
        
        if self.transform:
            image = self.transform(image)
        
        if "target" in self.df.columns.values:
            target = self.df.iloc[idx]["target"]
            return {"image": image,
                    "target": target}
        else:
            return {"image": image}
    
    def __len__(self):
        return len(self.df)


# In[ ]:


class ResizedNpyMelanomaDataset(Dataset):
    
    def __init__(self, npy_file, indices_to_select, df=None, transform=None):
        self.transform = transform
        self.npy_file = npy_file
        self.df = df
        self.indices_to_select = indices_to_select
    
    def __getitem__(self, n):
        idx = self.indices_to_select[n]
        
        image = Image.fromarray(self.npy_file[idx])
        if self.transform:
            image = self.transform(image)
        
        target = self.df.loc[idx].target
        
        return {"image": image,
                "target": target}
    
    def __len__(self):
        return len(self.indices_to_select)


# In[ ]:


class AlbuMelanomaDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if "target" in self.df.columns.values:
            target = self.df.iloc[idx]["target"]
            return {"image": image,
                    "target": target}
        else:
            return {"image": image}
    
    def __len__(self):
        return len(self.df)


# ## Augmentations <a class="anchor" id="augmentations"></a> 

# Many thanks to Roman for his notebook [Melanoma. Pytorch starter. EfficientNet](https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet). I liked the special augmentations very much and will use them here as well:

# In[ ]:


def random_microscope(img):
    circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

    mask = circle - 255
    img = np.multiply(img, mask)
    return img


# In[ ]:


import cv2

class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        img = np.asarray(img)
        if random.random() < p:
            img = random_microscope(img)
        img = Image.fromarray(np.uint8(img))
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

    
class AlbuMicroscope(ImageOnlyTransform):
    
    def __init__(self, always_apply=False, p=0.5):
        super(AlbuMicroscope, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        return random_microscope(img)


# In[ ]:


def transform_fun(resize_shape, key="train", plot=False):
    train_sequence = [transforms.Resize((resize_shape, resize_shape)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      Microscope(p=0.6)]
    dev_sequence = [transforms.Resize((resize_shape, resize_shape))]
    if plot==False:
        train_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dev_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    data_transforms = {'train': transforms.Compose(train_sequence),
                       'dev': transforms.Compose(dev_sequence),
                       'test_tta': transforms.Compose(train_sequence),
                       'test': transforms.Compose(dev_sequence)}
    return data_transforms[key]


# In[ ]:


def albu_transform_fun(resize_shape=None, key="train", plot=False):
    train_sequence = [
        Resize(resize_shape, resize_shape),
        RandomCrop(224,224),
        VerticalFlip(),
        HorizontalFlip(),
        RGBShift(r_shift_limit=40),
        RandomBrightness(0.1),
        AlbuMicroscope(p=0.6)]
    dev_sequence = [Resize(224, 224)]
    
    if plot==False:
        train_sequence.extend([
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],),
            ToTensor()])
        dev_sequence.extend([Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225],),
                             ToTensor()])
    
    data_transforms = {'train': Compose(train_sequence),
                       'dev': Compose(dev_sequence),
                       'test_tta': Compose(train_sequence),
                       'test': Compose(dev_sequence)}
    return data_transforms[key]


# Let's take a look at some example images and their augmented couterparts in train:

# In[ ]:


N = 10

fig, ax = plt.subplots(2,N,figsize=(20,5))

selection = np.random.choice(train_info.index.values, size=N, replace=False)

for n in range(N):
    
    org_image = cv2.imread(train_info.loc[selection[n]].image_path)
    org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    label = train_info.loc[selection[n]].target
    augmented = albu_transform_fun(resize_shape=256, key="train", plot=True)(**{"image":org_image, "label": label})
    ax[0,n].imshow(org_image)
    ax[1,n].imshow(augmented["image"])
    ax[1,n].axis("off")
    ax[0,n].axis("off")
    ax[0,n].set_title("Original")
    ax[1,n].set_title("Augmented");


# ## Loss and evaluation <a class="anchor" id="loss"></a> 

# ### Cross entropy loss
# 
# 
# $$L_{bce} = - \sum_{n}^{N} \sum_{k}^{2} t_{n,k} \cdot \log(y_{n,k}) = \sum_{n}^{N} \cdot l_{bce}$$
# 
# $$l_{bce} = - \sum_{k}^{2} t_{n,k} \cdot \log(y_{n,k}) $$

# In[ ]:


def get_ce_loss():   
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


# ### Weighted cross entropy loss
# 
# 
# $$L_{bce} = - \sum_{n}^{N} \sum_{k}^{2} \alpha_{k} \cdot t_{n,k} \cdot \log(y_{n,k}) $$

# In[ ]:


def get_wce_loss(train_targets):
    weights = compute_class_weight(y=train_targets,
                                   class_weight="balanced",
                                   classes=np.unique(train_targets))    
    class_weights = torch.FloatTensor(weights)
    if device.type=="cuda":
        class_weights = class_weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    return criterion


# ### Focal loss

# $$L_{focal} = - \sum_{n}^{N} \sum_{k}^{2} \alpha_{k} \cdot t_{n,k} \cdot (1-y_{n,k})^{\gamma} \cdot \log(y_{n,k})$$

# In[ ]:


class MulticlassFocalLoss(torch.nn.Module):
    
    def __init__(self, train_targets=None, gamma=2):
        super(MulticlassFocalLoss, self).__init__()
        self.gamma = gamma
        if train_targets is None:
            self.class_weights = None
        else:
            weights = compute_class_weight(y=train_targets,
                                   class_weight="balanced",
                                   classes=np.unique(train_targets))    
            self.class_weights = torch.FloatTensor(weights)
            if device.type=="cuda":
                self.class_weights = self.class_weights.cuda()
    
    def forward(self, input, target):
        if self.class_weights is None:
            ce_loss = F.cross_entropy(input, target, reduction='none')
        else:
            ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        loss = (1-pt)**self.gamma * ce_loss
        return torch.mean(loss)


# ## Model structure <a class="anchor" id="model"></a> 
# 
# Caution: I haven't implemented some way for densenet so far.

# In[ ]:


os.listdir(modelspath)


# In[ ]:


def get_model(kind="resnet34"):
    if kind == "resnet34":
        model = models.resnet34(pretrained=False)
        model.load_state_dict(torch.load(modelspath + "resnet34.pth"))
    elif kind == "resnet50":
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(modelspath + "resnet50.pth"))
    elif kind == "densenet121":
        model = models.densenet121(pretrained=False)
        model.load_state_dict(torch.load(modelspath + "densenet121.pth"))
    elif kind == "densenet201":
        model = models.densenet201(pretrained=False)
        model.load_state_dict(torch.load(modelspath + "densenet201.pth"))
    elif kind == "efficientnet_b1":
        model = EfficientNet.from_pretrained('efficientnet-b1')
    else:
        model = models.resnet34(pretrained=False)
        model.load_state_dict(torch.load(modelspath + "resnet34.pth"))
    return model        


# In[ ]:


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def build_model(the_model):
    model = get_model(the_model)
    
    if "efficientnet" in the_model:
        num_features = model._fc.in_features
    else:
        num_features = model.fc.in_features
    
    basic_modules = torch.nn.Sequential(torch.nn.Linear(num_features, 128),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(128),
                                        torch.nn.Dropout(0.2),

                                        torch.nn.Linear(128, num_classes))
    
    if "efficientnet" in the_model:
        model._fc = basic_modules
    else:
        model.fc = basic_modules
        
    
    return model


# ## Predict on whatever you like <a class="anchor" id="predict"></a>

# In[ ]:


# make sure that counter*batch_size is the same as len(dataset)
def predict(fold_results, dataloader, TTA=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    avg_preds = np.zeros(len(dataloader.dataset))
    avg_probas = np.zeros((len(dataloader.dataset),2))
        
    for fold_num in fold_results.keys():
        
        results = fold_results[fold_num]
        model = results.model
        
        dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))
        
        for t in range(TTA):
            print("TTA phase {}".format(t))
            for counter, data in enumerate(dataloader_iterator):    
                image_input = data["image"]
                image_input = image_input.to(device, dtype=torch.float)

                pred_probas = model(image_input)
                _, preds = torch.max(pred_probas, 1)

                avg_preds[
                    (counter*dataloader.batch_size):(dataloader.batch_size*(counter+1))
                ] += preds.cpu().detach().numpy()/(len(fold_results)*TTA)
                avg_probas[
                    (counter*dataloader.batch_size):(dataloader.batch_size*(counter+1))
                ] += softmax(pred_probas.cpu().detach().numpy(), axis=1)/(len(fold_results)*TTA)
        
    return avg_preds, avg_probas


# ## Searching for an optimal learning rate <a class="anchor" id="learning_rate_search"></a>

# In[ ]:


def get_scheduler(optimiser, min_lr, max_lr, stepsize):
    # suggested_stepsize = 2*num_iterations_within_epoch
    stepsize_up = np.int(stepsize/2)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser,
                                               base_lr=min_lr,
                                               max_lr=max_lr,
                                               step_size_up=stepsize_up,
                                               step_size_down=stepsize_up,
                                               mode="triangular")
    return scheduler
    


# In[ ]:


def get_lr_search_scheduler(optimiser, min_lr, max_lr, max_iterations):
    # max_iterations should be the number of steps within num_epochs_*epoch_iterations
    # this way the learning rate increases linearily within the period num_epochs*epoch_iterations 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser, 
                                               base_lr=min_lr,
                                               max_lr=max_lr,
                                               step_size_up=max_iterations,
                                               step_size_down=max_iterations,
                                               mode="triangular")
    
    return scheduler


# ## Training loop <a class="anchor" id="training_loop"></a>

# In[ ]:


from scipy.special import expit

def run_training(criterion,
                 num_epochs,
                 dataloaders_dict,
                 fold_num,
                 patience,
                 results,
                 find_lr):
    
    if find_lr:
        phases = ["train"]
    else:
        phases = ["train", "dev"]
        
    best_auc = 0
    patience_counter = 0
    epsilon = 1e-7
    
    for epoch in range(num_epochs):
        
        for phase in phases:
            
            dataloader = dataloaders_dict[phase]
            dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))
            
            if phase=="train":
                results.model.train()
            else:
                results.model.eval()
                
            all_preds = np.zeros(len(dataloader)*dataloader.batch_size)
            all_targets = np.zeros(len(dataloader)*dataloader.batch_size)   
            running_loss = 0.0
            running_true_positives = 0
            running_false_positives = 0
            running_false_negatives = 0
            
                      
            for counter, data in enumerate(dataloader_iterator):
                image_input = data["image"]
                target_input = data["target"]
                
                image_input = image_input.to(device, dtype=torch.float)
                target_input = target_input.to(device, dtype=torch.long)
    
                results.optimiser.zero_grad()
                
                raw_output = results.model(image_input) 
                
                _, preds = torch.max(raw_output,1)
                
                running_true_positives += (preds*target_input).sum().cpu().detach().numpy()
                running_false_positives += ((1-target_input)*preds).sum().cpu().detach().numpy()
                running_false_negatives += (target_input*(1-preds)).sum().cpu().detach().numpy()

                precision = running_true_positives / (running_true_positives + running_false_positives + epsilon)
                recall = running_true_positives / (running_true_positives + running_false_negatives + epsilon)
                
                f1_score = 2*precision*recall / (precision+recall+epsilon) 
                
                
                results.results[phase].learning_rates.append(optimiser.state_dict()["param_groups"][0]["lr"])
                results.results[phase].precision.append(precision)
                results.results[phase].recall.append(recall)
                results.results[phase].f1_scores.append(f1_score)
                        
                batch_size = dataloader.batch_size
                all_targets[(counter*batch_size):((counter+1)*batch_size)] = target_input.cpu().detach().numpy()
                all_preds[(counter*batch_size):((counter+1)*batch_size)] = preds.cpu().detach().numpy()
                
                loss = criterion(raw_output, target_input)
                # redo the average over mini_batch
                running_loss += (loss.item() * batch_size)
    
                # save averaged loss over processed number of batches:
                processed_loss = running_loss / ((counter+1) * batch_size)
                results.results[phase].losses.append(processed_loss)
                
                if phase == 'train':
                    loss.backward()
                    results.optimiser.step()
                    if results.scheduler is not None:
                        results.scheduler.step()
                        
            epoch_auc_score = roc_auc_score(all_targets, all_preds)
            results.results[phase].epoch_scores.append(epoch_auc_score)
                
            
            # average over all samples to obtain the epoch loss
            epoch_loss = running_loss / len(dataloader.dataset)
            results.results[phase].epoch_losses.append(epoch_loss)
            
            print("fold: {}, epoch: {}, phase: {}, e-loss: {}, e-auc: {}".format(
                fold_num, epoch, phase, epoch_loss, epoch_auc_score))
            
            if not find_lr:
                if phase == "dev":
                    if epoch_auc_score >= best_auc:
                        best_auc = epoch_auc_score
                        best_model_wts = copy.deepcopy(results.model.state_dict())
                        best_model_optimiser = copy.deepcopy(results.optimiser.state_dict())
                        best_scheduler = copy.deepcopy(results.scheduler.state_dict())
                        best_epoch = epoch
                        best_loss = processed_loss
                    else:
                        patience_counter += 1
                        if patience_counter == patience:
                            print("Model hasn't improved for {} epochs. Training finished.".format(patience))
                            break
               
    # load best model weights
    if not find_lr:
        results.model.load_state_dict(best_model_wts)
        results.optimiser.load_state_dict(best_model_optimiser)
        results.scheduler.load_state_dict(best_scheduler)
        results.best_epoch = best_epoch
        results.best_loss = best_loss
    return results


# In[ ]:


class ResultsBean:
    
    def __init__(self):
        
        self.precision = []
        self.recall = []
        self.f1_scores = []
        self.losses = []
        self.learning_rates = []
        self.epoch_losses = []
        self.epoch_scores = []

class Results:
    
    def __init__(self, fold_num, model=None, optimiser=None, scheduler=None, model_kind="resnet34"):
        self.model = model
        self.model_kind = model_kind
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.best_epoch = 0
        self.best_loss = 0
        
        self.fold_num = fold_num
        self.train_results = ResultsBean()
        self.dev_results = ResultsBean()
        self.results = {"train": self.train_results,
                        "dev": self.dev_results}


# In[ ]:


def train(model,
          model_kind,
          criterion,
          optimiser,
          num_epochs,
          dataloaders_dict,
          fold_num,
          scheduler,
          patience,
          find_lr=False):
    
    single_results = Results(fold_num=fold_num,
                             model=model,
                             optimiser=optimiser,
                             scheduler=scheduler,
                             model_kind=model_kind)
    
    
    single_results = run_training(criterion,
                                  num_epochs,
                                  dataloaders_dict,
                                  fold_num,
                                  patience,
                                  single_results, 
                                  find_lr=find_lr)
       
    return single_results


# In[ ]:


def save_as_csv(series, name, path):
    df = pd.DataFrame(index=np.arange(len(series)), data=series, columns=[name])
    output_path = path + name + ".csv"
    df.to_csv(output_path, index=False)

def save_results(results, foldername):
    for fold in results.keys():
        
        base_dir = foldername + "/fold_" + str(fold) + "/"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # save the model for inference
        model = results[fold].model
        model_kind = results[fold].model_kind
        #model_path = base_dir + model_kind + ".pth"
        #torch.save(model.state_dict(), model_path)
        
        # save checkpoint for inference and retraining:
        checkpoint_path = base_dir + model_kind + ".tar"
        torch.save({
            'epoch': results[fold].best_epoch,
            'loss': results[fold].best_loss,
            'model_state_dict': results[fold].model.state_dict(),
            'optimizer_state_dict': results[fold].optimiser.state_dict(),
            'scheduler_state_dict': results[fold].scheduler.state_dict()}, checkpoint_path)
        
        for phase in ["train", "dev"]:
            losses = results[fold].results[phase].losses
            epoch_losses = results[fold].results[phase].epoch_losses
            epoch_scores = results[fold].results[phase].epoch_scores
            lr_rates = results[fold].results[phase].learning_rates
            f1_scores = results[fold].results[phase].f1_scores
            precision = results[fold].results[phase].precision
            recall = results[fold].results[phase].recall
            
            save_as_csv(losses, phase + "_losses", base_dir)
            save_as_csv(epoch_losses, phase + "_epoch_losses", base_dir)
            save_as_csv(epoch_scores, phase + "_epoch_scores", base_dir)
            save_as_csv(lr_rates, phase + "_lr_rates", base_dir)
            save_as_csv(f1_scores, phase + "_f1_scores", base_dir)
            save_as_csv(precision, phase + "_precision", base_dir)
            save_as_csv(recall, phase + "_recall", base_dir)


# In[ ]:


def load_checkpoint(model_kind,
                    checkpoint_path,
                    for_inference,
                    single_results,
                    lr,
                    num_epochs,
                    min_lr, max_lr, len_train):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path)
    
    single_results.model = build_model(model_kind)
    single_results.model.load_state_dict(checkpoint["model_state_dict"])
    single_results.model.to(device)
    
    if "efficientnet" in model_kind:
        single_results.optimiser = torch.optim.SGD(single_results.model._fc.parameters(), lr=lr)
    else:
        single_results.optimiser = torch.optim.SGD(single_results.model.fc.parameters(), lr=lr)
    single_results.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    
    max_iterations = num_epochs * len_train
    single_results.scheduler = get_lr_search_scheduler(single_results.optimiser, min_lr, max_lr, max_iterations)
    single_results.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    single_results.best_epoch = checkpoint["epoch"]
    single_results.best_loss = checkpoint["loss"]
    
    # set into inference state
    if for_inference:
        single_results.model.eval()
    else:
        single_results.model.train()
    
    return single_results


# In[ ]:


def load_results(save_folder, total_folds, model_kind, lr, num_epochs, len_train, min_lr, max_lr, for_inference=True):
    results = {}
    
    for fold in range(total_folds): 
        single_results = Results(fold)
        
        base_path = save_folder + "/fold_" + str(fold) + "/"
        checkpoint_path = base_path + model_kind + ".tar"
        single_results = load_checkpoint(model_kind,
                                         checkpoint_path,
                                         for_inference,
                                         single_results,
                                         lr,
                                         num_epochs,
                                         min_lr, max_lr, len_train)
        
        for phase in ["train", "dev"]:
            single_results.results[phase].losses = pd.read_csv(base_path + phase + "_losses.csv")
            single_results.results[phase].epoch_losses = pd.read_csv(base_path + phase + "_epoch_losses.csv")
            single_results.results[phase].epoch_scores = pd.read_csv(base_path + phase + "_epoch_scores.csv")
            single_results.results[phase].learning_rates = pd.read_csv(base_path + phase + "_lr_rates.csv")
            single_results.results[phase].f1_scores = pd.read_csv(base_path + phase + "_f1_scores.csv")
            single_results.results[phase].precision = pd.read_csv(base_path + phase + "_precision.csv")
            single_results.results[phase].recall = pd.read_csv(base_path + phase + "_recall.csv")
        
        results[fold] = single_results
    return results


# # Experimental zone <a class="anchor" id="experimental_zone"></a>

# ## Creating a hold-out dataset <a class="anchor" id="holdout"></a>
# 
# We have observed a big missing group of test images that is not present in the training data. With ~15 % this should have a big impact on the score-differences between CV-scores and the leaderboard. To find strategies to overcome this problem we could exclude a small common group of the training data and use their images within a hold-out dataset. Furthermore we should also use a part of the remaining images to cover all other training groups in the hold-out as well. 

# In[ ]:


train_image_stats.head(1)


# In[ ]:


train_image_stats.groupby(
    ["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / train_image_stats.shape[0]


# In[ ]:


relevant_groups = train_image_stats.groupby(
    ["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10].index.values


# Ok, let's pick the group of 480 rows and 640 columns or the group with 3456 rows and 5184 columns. They are big enough to simulate what will happen when they are present in the hold-out data but not in train. To get a feeling how different their image information may look like from the other groups, we could plot the median of image means and stds of all groups and our candidates:

# In[ ]:


group_stats = pd.DataFrame(train_image_stats.groupby(["rows", "columns"]).img_mean.median().loc[relevant_groups])
group_stats["img_std"] = train_image_stats.groupby(["rows", "columns"]).img_std.median().loc[relevant_groups]


# In[ ]:


plt.figure(figsize=(20,5))
plt.scatter(group_stats.img_mean, group_stats.img_std, label="remaining train groups");
plt.scatter(group_stats.loc[(480, 640)].img_mean, group_stats.loc[(480, 640)].img_std,
            c="lime", label="candidate 480, 640")
plt.scatter(group_stats.loc[(3456, 5184)].img_mean, group_stats.loc[(3456, 5184)].img_std,
            c="deeppink", label="candidate 3456, 5184");
plt.title("Image statistics groups");
plt.legend()
plt.xlabel("Median of group image means")
plt.ylabel("Std of group image means");


# Ok, both groups look somehow far away from the others. Let's take the smaller one: 3456, 5184.

# In[ ]:


selected_hold_out_group = train_image_stats.loc[
    (train_image_stats["rows"]==3456) & (train_image_stats["columns"]==5184)
].path.values

hold_out_df = train_info.loc[train_info.dcm_path.isin(selected_hold_out_group)].copy()
reduced_train_df = train_info.loc[train_info.dcm_path.isin(selected_hold_out_group)==False].copy()


# In[ ]:


hold_out_df.shape[0] / reduced_train_df.shape[0]


# Ok, now we have created a hold out dataset that only consists of one type of image group. We need to fill it up with further training samples of all other groups. To be similar to the test set we should try to reach a 33% split.  

# In[ ]:


test_info.shape[0] / train_info.shape[0]


# In[ ]:


reduced_train_df, add_to_hold_out_df = train_test_split(
    reduced_train_df, test_size=0.163, stratify=reduced_train_df.target.values)


# In[ ]:


hold_out_df = hold_out_df.append(add_to_hold_out_df)
print(hold_out_df.shape[0] / reduced_train_df.shape[0])
print(hold_out_df.shape[0], reduced_train_df.shape[0])


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))

h_target_perc = hold_out_df.target.value_counts() / hold_out_df.shape[0] * 100
rt_target_perc = reduced_train_df.target.value_counts() / reduced_train_df.shape[0] * 100 

sns.barplot(h_target_perc.index, h_target_perc.values, ax=ax[0], palette="Oranges_r")
sns.barplot(rt_target_perc.index, rt_target_perc.values, ax=ax[1], palette="Purples_r");

ax[0].set_title("Target distribution of \n hold-out");
ax[1].set_title("Target distribution of \n reduced train");
for n in range(2):
    ax[n].set_ylabel("% in data")
    ax[n].set_xlabel("Target")


# Both have very similar target distributions now:

# In[ ]:


h_target_perc


# In[ ]:


rt_target_perc


# ## Settings <a class="anchor" id="settings"></a>

# Let's now chose a model structure:

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


torch.manual_seed(0)
np.random.seed(0)


# Let's start with resnet:

# In[ ]:


num_classes = 2

my_model = "resnet50"
TRAIN_BATCH_SIZE = 64
LR = 0.01


# ## Using Bojans resized original images to speed up <a class="anchor" id="using_resized"></a>

# And let's pick a resize shape of Bojan Tunguz resized images:

# In[ ]:


os.listdir("../input/siimisic-melanoma-resized-images")


# This way loading the images will be much faster than doing this on the fly.

# In[ ]:


RESIZE_SHAPE = 256


# In[ ]:


#train_npy = np.load("../input/siimisic-melanoma-resized-images/x_train_" + str(RESIZE_SHAPE) + ".npy", mmap_mode="r")
#train_npy.shape


# In[ ]:


#x_test = np.load("../input/siimisic-melanoma-resized-images/x_test_" + str(RESIZE_SHAPE) + ".npy", mmap_mode="r")
#x_test.shape


# As we are using a hold-out dataset to simulate what happens when there is a image group in test that is missed in train, we need to selected the proper indices:

# In[ ]:


hold_out_indices = hold_out_df.index.values
reduced_train_indices = reduced_train_df.index.values


# In[ ]:


#hold_out_dataset_1 = ResizedNpyMelanomaDataset(train_npy, hold_out_indices, df=hold_out_df,
#                                              transform=transform_fun(RESIZE_SHAPE, key="dev", plot=True))
#hold_out_dataset_2 = MelanomaDataset(hold_out_df, transform=transform_fun(RESIZE_SHAPE, key="dev", plot=True))


# In[ ]:


#idx = 10

#hold_out_example_1 = hold_out_dataset_1.__getitem__(idx)
#hold_out_example_2 = hold_out_dataset_2.__getitem__(idx)

#fig, ax = plt.subplots(1,4,figsize=(20,5))
#ax[0].imshow(hold_out_example_1["image"])
#ax[0].axis("off")
#ax[0].set_title(hold_out_example_1["target"]);
#sns.distplot(hold_out_example_1["image"], ax=ax[1])
#ax[2].imshow(hold_out_example_2["image"])
#ax[2].axis("off")
#ax[2].set_title(hold_out_example_2["target"]);
#sns.distplot(hold_out_example_1["image"], ax=ax[3])


# Doing the resize-preprocessing in advance is definitely speeding up the computation! If you are using more images than the original data you should consider to do so as well.

# In[ ]:


train_indices = train_info.index.values


# ## Searching for learning rate boundaries <a class="anchor" id="lr_bounds"></a>
# 
# Our first task would be to find proper learning rate boundaries to use the cyclical learning rate approach. I'm following the triangular method described in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf).
# 
# 
# *More explanations are following soon*
# 
# 
# #### But how to find the best min and max learning rates?
# 
# * Within a predefined number of epochs the learning rates is increased linearily between to boundary values of your choise: min_lr and max_lr. 
# * While training I'm currently measuring the running f1_score on train data to observe how this increasing rate changes the quality of predictions.
# * I'm not using accuracy score as our target distribution is highly imbalanced and it's easy to get high accuracy scores by only predicting lots of zeros. 

# If you like to search for optimal min and max learning rates, just choose your values and set find_lr=True. The results of the search will be saved in the save_folder specified. If you like you can add your result as a dataset and specify their path in load_folder. This way you can visualize them in the plot that follows the search.

# In[ ]:


find_lr = True
min_lr = 0.001
max_lr = 1
NUM_EPOCHS = 3
save_folder = "learning_rate_search"
load_folder = "../input/melanomaclassificationsmoothiestarter/learning_rate_search"


# At the moment I'm using this [external data ](https://www.kaggle.com/nroman/melanoma-external-malignant-256) as I found it difficult to train without more positive cases. Consequently the idea of predicting on a hold-out group should be done on external data. I haven't done this yet and I probably won't find enough time to redo everything on external data. This is up to you! ;-)

# In[ ]:


external_data_path = "../input/melanoma-external-malignant-256/"
external_train = pd.read_csv(external_data_path + "/train_concat.csv")
external_train["image_path"] = external_data_path + "train/train/" + external_train.image_name + ".jpg"


# Let's take a look at the different target distributions:

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train_info.target, ax=ax[0], palette="Reds_r")
sns.countplot(external_train.target, ax=ax[1], palette="Reds_r")
ax[1].set_title("Target imbalance in external train")
ax[0].set_title("Target imbalance in original train");


# Ok, let's try to find good learning rate boundaries:

# In[ ]:


if find_lr:
    
    results = {}
    
    #train_idx, dev_idx = train_test_split(train_indices,
                                          #stratify=train_info.target.values,
                                          #test_size=0.3,
                                          #random_state=0)
    
    #train_dataset = ResizedNpyMelanomaDataset(npy_file=train_npy,
    #                                          indices_to_select=train_idx, 
    #                                          df=train_info,
    #                                          transform=transform_fun(RESIZE_SHAPE, "train"))
    #dev_dataset = ResizedNpyMelanomaDataset(npy_file=train_npy,
    #                                        indices_to_select=dev_idx, 
    #                                        df=train_info,
    #                                        transform=transform_fun(RESIZE_SHAPE, "dev"))
    
    
    train_df, dev_df = train_test_split(external_train,
                                        stratify=external_train.target.values,
                                        test_size=0.3,
                                        random_state=0)
    
    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))
    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}
    
    model = build_model(my_model)
    model.apply(init_weights)
    model = model.to(device)
    
    # if you are using the external data
    criterion = MulticlassFocalLoss(gamma=2)
    # if you are using the resized data:
    #criterion = MulticlassFocalLoss(train_info.iloc[train_idx].target.values)
    #criterion = get_wce_loss(train_df.target.values)
    
    if "efficientnet" in my_model:
        optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)
    else:
        optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)
    
    max_iterations = NUM_EPOCHS * len(train_dataloader)
    scheduler = get_lr_search_scheduler(optimiser, min_lr, max_lr, max_iterations)
    
    single_results = train(model=model,
                           model_kind=my_model,
                           criterion=criterion,
                           optimiser=optimiser,
                           num_epochs=NUM_EPOCHS,
                           dataloaders_dict=dataloaders_dict,
                           fold_num=0,
                           scheduler=scheduler, 
                           patience=1,
                           find_lr=find_lr)
    
    results = {0: single_results}
    save_results(results, save_folder)
    
# prepare for retraining and/or inference:
else:
    train_df, dev_df = train_test_split(external_train,
                                        stratify=external_train.target.values,
                                        test_size=0.3,
                                        random_state=0)
    
    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))
    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}
    
    criterion = MulticlassFocalLoss(gamma=2)
    
    results = load_results(load_folder,
                           total_folds=1,
                           model_kind=my_model,
                           lr=LR,
                           num_epochs=NUM_EPOCHS,
                           len_train=len(train_dataloader),
                           min_lr=min_lr,
                           max_lr=max_lr,
                           for_inference=True)


# Personally I find it a bit easier to use weighted cross entropy loss but perhaps with tuning the hyperparameters properly the focal loss could be a good choice as well. Try to start with $\gamma=0$ as the focal loss would turn into weighted cross entropy loss in this case. I'm also working on a better way to set $\alpha$. The class weights are not working well and there seems to be a way to set them based on the effective number of samples ([look at this paper](https://arxiv.org/abs/1901.05555)). I would like to try it out. 

# In[ ]:


fig, ax = plt.subplots(3,2,figsize=(20,18))

rates = results[0].results["train"].learning_rates
f1_score = results[0].results["train"].f1_scores
precision = results[0].results["train"].precision
recall = results[0].results["train"].recall
losses = results[0].results["train"].losses
epoch_losses = results[0].results["train"].epoch_losses

ax[0,0].plot(rates, f1_score, '.-', c="maroon", label="f1-score")
ax[0,0].plot(rates, precision, '.-', c="salmon", label="precision")
ax[0,0].plot(rates, recall, '.-', c="lightsalmon", label="recall")


ax[0,0].legend();
ax[0,0].set_xlabel("Learning rate")
ax[0,0].set_ylabel("Score values")
ax[0,0].set_title("Evaluation scores for learning rate search within {} epochs".format(NUM_EPOCHS));

ax[1,1].plot(rates, precision, '.-', c="salmon", label="precision")
ax[1,1].set_title("Precision")
ax[1,1].set_xlabel("learning rates")
ax[1,1].set_ylabel("precision")

ax[1,0].plot(rates, recall, '.-', c="lightsalmon", label="recall")
ax[1,0].set_title("Recall")
ax[1,0].set_xlabel("learning rates")
ax[1,0].set_ylabel("recall")

ax[0,1].plot(rates, f1_score, '.-', c="maroon", label="f1-score")
ax[0,1].set_title("F1-score")
ax[0,1].set_xlabel("learning rates")
ax[0,1].set_ylabel("f1-score")

ax[2,0].plot(rates, losses, 'o-', c="deepskyblue")
ax[2,0].set_title("Loss change with rates")
ax[2,0].set_ylabel("loss")
ax[2,0].set_xlabel("Learning rates")

ax[2,1].set_title("Learning rate increase")
ax[2,1].plot(rates, 'o', c="mediumseagreen");
ax[2,1].set_ylabel("learning rate")
ax[2,1].set_xlabel("Iteration step");


# ### Insights
# 
# * You may like to play with the smallest learning rate first. During my experiments I found that how small the first one is definitely influences the success of these curves and how much you can increase the max learning rate.
# * One can also see that the loss may go up and down a bit even though the scores are still increasing!

# ## Running a model <a class="anchor" id="running"></a>
# 
# Let's now check wheater everything works as expected:

# In[ ]:


check_workflow = False
save_folder = "check_workflow"
load_folder = "../input/melanomaclassificationsmoothiestarter/check_workflow"
NUM_EPOCHS = 10
LR = 0.01
min_lr = 0.0001
max_lr = 0.25
find_lr=False


# In[ ]:


if check_workflow:
    
    results = {}
    
    train_df, dev_df = train_test_split(external_train,
                                        stratify=external_train.target.values,
                                        test_size=0.3,
                                        random_state=0)
    
    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))
    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}
    
    model = build_model(my_model)
    model.apply(init_weights)
    model = model.to(device)
    
    criterion = MulticlassFocalLoss(gamma=2)
    #criterion = get_wce_loss(train_df.target.values)
    if "efficientnet" in my_model:
        optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)
    else:
        optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)
    
    stepsize = 2*len(train_dataloader)
    scheduler = get_scheduler(optimiser, min_lr, max_lr, stepsize)
    
    single_results = train(model=model,
                           model_kind=my_model,
                           criterion=criterion,
                           optimiser=optimiser,
                           num_epochs=NUM_EPOCHS,
                           dataloaders_dict=dataloaders_dict,
                           fold_num=0,
                           scheduler=scheduler, 
                           patience=1,
                           find_lr=find_lr)
    
    results = {0: single_results}
    save_results(results, save_folder)

else:
    
    train_df, dev_df = train_test_split(external_train,
                                        stratify=external_train.target.values,
                                        test_size=0.3,
                                        random_state=0)
    
    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))
    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}
    
    criterion = MulticlassFocalLoss(gamma=2)
    
    results = load_results(load_folder,
                           total_folds=1,
                           model_kind=my_model,
                           lr=LR,
                           num_epochs=NUM_EPOCHS,
                           len_train=len(train_dataloader),
                           min_lr=min_lr,
                           max_lr=max_lr,
                           for_inference=True)


# In[ ]:


save_results(results, save_folder)


# In[ ]:


fig, ax = plt.subplots(3,2,figsize=(20,15))

rates = results[0].results["train"].learning_rates
f1_score = results[0].results["train"].f1_scores
precision = results[0].results["train"].precision
recall = results[0].results["train"].recall

train_losses = results[0].results["train"].losses
dev_losses = results[0].results["dev"].losses

train_epoch_losses = results[0].results["train"].epoch_losses
dev_epoch_losses = results[0].results["dev"].epoch_losses
train_epoch_auc = results[0].results["train"].epoch_scores
dev_epoch_auc = results[0].results["dev"].epoch_scores

ax[0,0].plot(f1_score, '.-', c="maroon", label="f1-score")
ax[0,0].plot(precision, '.-', c="salmon", label="precision")
ax[0,0].plot(recall, '.-', c="lightsalmon", label="recall")

ax[0,0].legend();
ax[0,0].set_xlabel("Learning rate")
ax[0,0].set_ylabel("Score values")
ax[0,0].set_title("Evaluation scores for learning rate search within {} epochs".format(NUM_EPOCHS));

ax[0,1].plot(rates)
ax[0,1].set_title("Learning rates")

ax[1,0].plot(train_losses, label="train")

ax[1,1].plot(dev_losses, label="dev");
ax[1,1].legend()
ax[1,1].set_title("Losses")

ax[2,0].plot(train_epoch_losses, label="train")
ax[2,0].plot(dev_epoch_losses, label="dev")
ax[2,0].set_title("Epoch losses")

ax[2,1].plot(train_epoch_auc)
ax[2,1].plot(dev_epoch_auc)
ax[2,1].set_title("Epoch AUC");


# ## Running models with StratifiedKFold

# In[ ]:


run_kfold = False
n_splits = 3
save_folder = "kfold_workflow"
load_folder = "../input/melanomaclassificationsmoothiestarter/kfold_workflow"
NUM_EPOCHS = 10
LR = 0.01
min_lr = 0.0001
max_lr = 0.25
find_lr=False


# In[ ]:


skf = StratifiedKFold(n_splits=5, random_state=0)


# In[ ]:


if run_kfold:
    
    results = {}
    
    n_fold = 0
    for train_idx, dev_idx in skf.split(external_train, external_train.target.values):
        train_df = external_train.iloc[train_idx]
        dev_df = external_train.iloc[dev_idx]
        
    
        train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))
        dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
    
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
        dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}

        model = build_model(my_model)
        model.apply(init_weights)
        model = model.to(device)
        
        criterion = MulticlassFocalLoss(gamma=2)
        #criterion = get_wce_loss(train_df.target.values)
        if "efficientnet" in my_model:
            optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)
        else:
            optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)
    
        stepsize = 2*len(train_dataloader)
        scheduler = get_scheduler(optimiser, min_lr, max_lr, stepsize)
    
        single_results = train(model=model,
                               model_kind=my_model,
                               criterion=criterion,
                               optimiser=optimiser,
                               num_epochs=NUM_EPOCHS,
                               dataloaders_dict=dataloaders_dict,
                               fold_num=0,
                               scheduler=scheduler, 
                               patience=1,
                               find_lr=find_lr)
    
        results = {n_fold: single_results}
        n_fold += 1
    
    save_results(results, save_folder)


# ## Exploring predictions and weaknesses <a class="anchor" id="result_analysis"></a>

# Let's use the last dev dataset to yield some insights about predictions and weaknesses of our model:

# In[ ]:


max_size = 120

for m in range(max_size+1):
    to_try = max_size - m
    if dev_df.shape[0] % to_try == 0:
        break
        
DEV_BATCH_SIZE = to_try
to_try


# ### Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

def get_confusion_matrix(y_true, y_pred):
    transdict = {1: "malignant", 0: "benign"}
    y_t = np.array([transdict[x] for x in y_true])
    y_p = np.array([transdict[x] for x in y_pred])
    
    labels = ["benign", "malignant"]
    index_labels = ["actual benign", "actual malignant"]
    col_labels = ["predicted benign", "predicted malignant"]
    confusion = confusion_matrix(y_t, y_p, labels=labels)
    confusion_df = pd.DataFrame(confusion, index=index_labels, columns=col_labels)
    for n in range(2):
        confusion_df.iloc[n] = confusion_df.iloc[n] / confusion_df.sum(axis=1).iloc[n]
    return confusion_df


# In[ ]:


dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))
dev_dataloader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, drop_last=False)
preds, probas = predict(results, dev_dataloader)


# In[ ]:


confusion = get_confusion_matrix(dev_df.target.values, preds)
plt.figure(figsize=(6,6))
sns.heatmap(confusion, cbar=False, annot=True, fmt="g", square=True, cmap="Reds");


# ## Submission <a class="anchor" id="submission"></a>

# In[ ]:


external_test_path = "../input/melanoma-external-malignant-256/test/test/"
test_info["image_path"] = external_test_path + test_info.image_name +".jpg"


# In[ ]:


TEST_BATCH_SIZE=68
test_dataset = AlbuMelanomaDataset(test_info, albu_transform_fun(RESIZE_SHAPE, "test"))
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)
preds, probas = predict(results, test_dataloader)


# In[ ]:


submission = pd.read_csv(basepath + "sample_submission.csv")
submission.target = probas[:,1]


# In[ ]:


submission.head()


# In[ ]:


sns.distplot(submission.target)


# In[ ]:


submission.to_csv("submission.csv", index=False)


# # Conclusion <a class="anchor" id="conclusion"></a>

# ## TODO

# 1. Add efficientnet as option (almost done)
# 2. Make retraining for best models possible (almost done)
#     * save all state dicts for model, optimizer, scheduler (done)
#     * show inference (done) and retraining of the model (todo)
#     * show updated losses and scores after retraining (todo)
# 3. Add stratified k-fold as validation scheme + oof to csv
# 8. Add more explanations 
# 9. Write a conclusion
