#!/usr/bin/env python
# coding: utf-8

# # SIIM-ISIC Melanoma Classification
# 
# In this competition, we have been tasked with identifying melanoma in images of skin lesions. In particular, images within the same patient can be used to determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.
# 
# Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.
# 

# ## Exploratory Data Analysis

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from skimage.transform import resize
import random
import os

print("List of directories:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


# In[ ]:


test_dir = '/kaggle/input/siim-isic-melanoma-classification/test'
train_dir = '/kaggle/input/siim-isic-melanoma-classification/train'
test_images = []
train_images = []
for dirname, _, filenames in os.walk(test_dir):
    for filename in filenames:
        test_images.append(filename)
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        train_images.append(filename)
print(f"The number of train images are: {len(train_images)}")
print(f"The number of test images are: {len(test_images)}")


# ### Visualizing DICOM images
# 
# 

# In[ ]:


# Checking the metadata of one DICOM image from the training set
image_number = random.randint(0, len(train_images))
image_dcm = pydicom.dcmread(os.path.join(train_dir, train_images[image_number]))
image_dcm


# In[ ]:


# Example image from training set
image_example = image_dcm.pixel_array
print(f"Shape of image: {image_example.shape}")
plt.imshow(image_example, interpolation='nearest')
plt.axis('off')


# #### Train images

# In[ ]:


# Let's look at multiple images from the training set resized to represent what will be input to a model e.g. ResNet50
IMG_SIZE = 224
selected_images = random.sample(range(len(train_images)), 16)
fig, ax = plt.subplots(4, 4, figsize=(20, 16))
ax = ax.flatten()
for a, i in zip(range(16), selected_images):
    dcm = pydicom.dcmread(os.path.join(train_dir, train_images[i]))
    image = dcm.pixel_array
    resized_img = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[a].imshow(resized_img, interpolation='nearest')
    ax[a].axis('off')
    ax[a].set_title('Image #{}'.format(i))


# In[ ]:


# The intensity distributions for these images
fig, ax = plt.subplots(4, 4, figsize=(20, 16))
ax = ax.flatten()
for a, i in zip(range(16), selected_images):
    dcm = pydicom.dcmread(os.path.join(train_dir, train_images[i]))
    image = dcm.pixel_array
    resized_img = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[a].hist(resized_img[:,:,0].ravel(), bins=50, color='red', alpha=0.3)
    ax[a].hist(resized_img[:,:,1].ravel(), bins=50, color='green', alpha=0.3)
    ax[a].hist(resized_img[:,:,2].ravel(), bins=50, color='blue', alpha=0.3)
    ax[a].set_xlim((0,1))
    ax[a].set_title('Image #{}'.format(i))


# #### Test images

# In[ ]:


IMG_SIZE = 224
selected_images = random.sample(range(len(test_images)), 16)
fig, ax = plt.subplots(4, 4, figsize=(20, 16))
ax = ax.flatten()
for a, i in zip(range(16), selected_images):
    dcm = pydicom.dcmread(os.path.join(test_dir, test_images[i]))
    image = dcm.pixel_array
    resized_img = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[a].imshow(resized_img, interpolation='nearest')
    ax[a].axis('off')
    ax[a].set_title('Image #{}'.format(i))


# In[ ]:


# The intensity distributions for these images
fig, ax = plt.subplots(4, 4, figsize=(20, 16))
ax = ax.flatten()
for a, i in zip(range(16), selected_images):
    dcm = pydicom.dcmread(os.path.join(test_dir, test_images[i]))
    image = dcm.pixel_array
    resized_img = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True) 
    ax[a].hist(resized_img[:,:,0].ravel(), bins=50, color='red', alpha=0.5)
    ax[a].hist(resized_img[:,:,1].ravel(), bins=50, color='green', alpha=0.5)
    ax[a].hist(resized_img[:,:,2].ravel(), bins=50, color='blue', alpha=0.5)
    ax[a].set_xlim((0,1))
    ax[a].set_title('Image #{}'.format(i))


# ### Analyzing train and set datasets

# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
print(f"Train data: {train.shape}\nTest data: {test.shape}")


# #### Training dataset

# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.patient_id.nunique()


# So, there are 33126 images for 2056 patients.

# In[ ]:


print(train.sex.value_counts())
print(train.sex.value_counts(normalize=True))


# In[ ]:


train['images_per_patient'] = train.groupby('patient_id')['patient_id'].transform('count')
print(f"The number of images per patient is in the range {train.images_per_patient.min()}:{train.images_per_patient.max()}")


# In[ ]:


plt.figure(figsize=(12,6))
train['age_approx'].value_counts().sort_index().plot(kind='bar')


# In[ ]:


train['diagnosis'].value_counts()


# In[ ]:


train['benign_malignant'].value_counts()


# In[ ]:


print(pd.crosstab(train['benign_malignant'], train['sex']))
print("\n")
print(pd.crosstab(train['benign_malignant'], train['sex'], normalize='columns'))


# In[ ]:


age_mal = train['age_approx'][train.benign_malignant=="malignant"].value_counts().sort_index()
age_mal_prop = train['age_approx'][train.benign_malignant=="malignant"].value_counts(normalize=True).sort_index()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].bar(age_mal.index,age_mal.values, width=3)
ax[0].set_xlabel('Approx age of patients')
ax[0].set_ylabel('Number of malignancies')
ax[1].bar(age_mal_prop.index,age_mal_prop.values, width=3)
ax[1].set_xlabel('Approx age of patients')
ax[1].set_ylabel('Proportion of malignancies')


# In[ ]:


print(pd.crosstab(train['benign_malignant'], train['age_approx'], normalize='columns'))


# For patients above the age of 60 there is at least a 2% chance that the image contains a malignant diagnosis.

# In[ ]:


train.anatom_site_general_challenge.value_counts()


# In[ ]:


print(pd.crosstab(train['benign_malignant'], train['anatom_site_general_challenge'], normalize='columns'))


# In[ ]:


pos_mal = train['anatom_site_general_challenge'][train.benign_malignant=="malignant"].value_counts()
pos_mal_prop = train['anatom_site_general_challenge'][train.benign_malignant=="malignant"].value_counts(normalize=True)
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].bar(pos_mal.index,pos_mal.values)
ax[0].set_xlabel('Anatomical location')
ax[0].set_ylabel('Number of malignancies')
ax[1].bar(pos_mal_prop.index,pos_mal_prop.values)
ax[1].set_xlabel('Anatomical location')
ax[1].set_ylabel('Proportion of malignancies')


# In[ ]:


new_cols = ['unknown', 'nevus', 'melanoma', 'seborrheic keratosis', 'lentigo NOS', 'lichenoid keratosis', 'solar lentigo', 'cafe-au-lait macule', 
            'atypical melanocytic proliferation']
for col in new_cols:
    train[col] = np.where(train['diagnosis']==col, 1, 0)


# In[ ]:


train.tail(10)


# In[ ]:


train.melanoma.value_counts()


# In[ ]:


train['conditions_per_patient'] = 0
for i, pid in enumerate(train.patient_id.unique()):
    cond = 0
    for col in new_cols[1:]:
        cond += np.where(train.loc[train['patient_id'] == pid, col].sum(axis=0) == 0, 0, 1)
    train.loc[train['patient_id'] == pid, 'conditions_per_patient'] = cond


# In[ ]:


train.head()


# In[ ]:


train.groupby('patient_id').first()['conditions_per_patient'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Number of conditions')
plt.ylabel('Number of patients')


# In[ ]:


train.groupby('patient_id').first()['conditions_per_patient'].value_counts().sort_index()


# In[ ]:


mel = [image_name + '.jpg' for image_name in train[train.melanoma==1].image_name]
nonmel = [image_name + '.jpg' for image_name in train[train.melanoma==0].image_name]


# In[ ]:


jpg_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
jpg_dir_test = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test'


# In[ ]:


# Random image
num = random.randint(0, len(mel))
image_jpg = plt.imread(os.path.join(jpg_dir, mel[num]))
plt.imshow(image_jpg, interpolation='nearest')
plt.axis('off')


# In[ ]:


# Random images selected for melanoma and without melanoma
num_images = 4
mel_images = random.sample(range(len(mel)), num_images)
nonmel_images = random.sample(range(len(mel)), num_images)
fig, ax = plt.subplots(2, 4, figsize=(16, 9))
ax = ax.flatten()
for i in range(num_images):
    img = plt.imread(os.path.join(jpg_dir, mel[mel_images[i]]))
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[i].imshow(img, interpolation='nearest')
    ax[i].axis('off')
    ax[i].set_title('Melanoma = 1')
for i in range(num_images):
    img = plt.imread(os.path.join(jpg_dir, nonmel[nonmel_images[i]]))
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[i+4].imshow(img, interpolation='nearest')
    ax[i+4].axis('off')
    ax[i+4].set_title('Melanoma = 0')


# In[ ]:


fig, ax = plt.subplots(2, 4, figsize=(20, 12))
ax = ax.flatten()
for i in range(num_images):
    img = plt.imread(os.path.join(jpg_dir, mel[mel_images[i]]))
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[i].hist(img[:,:,0].ravel(), bins=50, color='red', alpha=0.3)
    ax[i].hist(img[:,:,1].ravel(), bins=50, color='green', alpha=0.3)
    ax[i].hist(img[:,:,2].ravel(), bins=50, color='blue', alpha=0.3)
    ax[i].set_xlim((0,1))
    ax[i].set_title('Melanoma = 1')
for i in range(num_images):
    img = plt.imread(os.path.join(jpg_dir, nonmel[nonmel_images[i]]))
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[i+4].hist(img[:,:,0].ravel(), bins=50, color='red', alpha=0.3)
    ax[i+4].hist(img[:,:,1].ravel(), bins=50, color='green', alpha=0.3)
    ax[i+4].hist(img[:,:,2].ravel(), bins=50, color='blue', alpha=0.3)
    ax[i+4].set_xlim((0,1))
    ax[i+4].set_title('Melanoma = 0')


# #### Test dataset

# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.patient_id.nunique()


# There are 10982 images for 690 patients in the test set

# In[ ]:


print(test.sex.value_counts())
print(test.sex.value_counts(normalize=True))


# In[ ]:


test['images_per_patient'] = test.groupby('patient_id')['patient_id'].transform('count')
print(f"The number of images per patient is in the range {test.images_per_patient.min()}:{test.images_per_patient.max()}")


# In[ ]:


plt.figure(figsize=(12,6))
test['age_approx'].value_counts().sort_index().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(8,6))
test.anatom_site_general_challenge.value_counts().plot(kind='bar')
plt.ylabel('Number of images')


# In[ ]:


# Let's look at a small sample of test images with their intensity distributions
img_test = [image_name + '.jpg' for image_name in test.image_name]
jpg_test_images = random.sample(range(len(test_images)), num_images)
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
ax = ax.flatten()
for a, i in zip(range(num_images), jpg_test_images):
    img = plt.imread(os.path.join(jpg_dir_test, img_test[i]))
    resized_img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[a].imshow(resized_img, interpolation='nearest')
    ax[a].axis('off')
    ax[a].set_title('Image #{}'.format(i))
for a, i in zip(range(num_images), jpg_test_images):
    img = plt.imread(os.path.join(jpg_dir_test, img_test[i]))
    resized_img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    ax[a+4].hist(resized_img[:,:,0].ravel(), bins=50, color='red', alpha=0.3)
    ax[a+4].hist(resized_img[:,:,1].ravel(), bins=50, color='green', alpha=0.3)
    ax[a+4].hist(resized_img[:,:,2].ravel(), bins=50, color='blue', alpha=0.3)
    ax[a+4].set_title('Image #{}'.format(i))


# * The gender ratio is different in the test set (57:43) compared to the train set (52:48)
# * The age distribution appears to be similar
# * The anatomical site has the same sequence when sorted
# * On the whole, the test set has similar distributions of variables as in the train set
