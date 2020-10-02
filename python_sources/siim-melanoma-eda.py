#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


BASE_DIR = '/kaggle/input/siim-isic-melanoma-classification/'
# Location of the image dir
img_dir = BASE_DIR + 'jpeg/train/'


# In[ ]:


#imports
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().system('ls -lrt /kaggle/input/siim-isic-melanoma-classification')


# # Loading Data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# In[ ]:


train_df.head()


# # Initial EDA on train and test

# In[ ]:


# Check missing values on each column
train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


def print_value_counts_columnwise(df, listOfColumns):
    for col in listOfColumns:
        print(df[col].value_counts())


# In[ ]:


# Print count of unique values
print_value_counts_columnwise(train_df, ['sex','age_approx', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant','target'])


# Before plotting the distribution, let's fill all missing values

# In[ ]:


train_df.sex.fillna('na', inplace=True)
train_df.age_approx.fillna('na', inplace=True)
train_df.anatom_site_general_challenge.fillna('na', inplace=True)

#now check if there are still missing values in any column
train_df.isna().sum()


# In[ ]:


# Plot frequencies for each column
def freq_plots(cols):
    plt.figure(figsize = (15,10))
    for i in range(len(cols)):
        plt.subplot(2,3,i+1)
        plot = sns.countplot(x = cols[i], data = train_df)
        plt.xticks(rotation=45, horizontalalignment='right')


# In[ ]:


freq_plots(['sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant','target'])


# # Relative frequencies to check how balanced the distribution is!

# In[ ]:


# Plot relative frequencies for each column
def rel_freq_plots(cols):
    plt.figure(figsize = (15,10))
    for i in range(len(cols)):
        value_counts = train_df[cols[i]].value_counts(normalize=True)*100
        plt.subplot(2,3,i+1)
        plot = sns.barplot(x = value_counts.index, y = value_counts.values, alpha=0.8)
        plt.ylabel('Percentage %')
        plt.xticks(rotation=45, horizontalalignment='right')


# In[ ]:


rel_freq_plots(['sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant','target'])


# # Let's check how target is related to each feature

# In[ ]:


print(f"Out of {train_df.shape[0]} samples, we have {sum(train_df['target']==1)} malignant and {sum(train_df['target']==0)} benign cases, which leads to {sum(train_df['target']==1)/train_df.shape[0] * 100} % positive class and    {sum(train_df['target']==0)/train_df.shape[0] * 100} % negative class")


# In[ ]:


# check how target is distributed on sex
tmp = train_df.groupby('sex')['target'].value_counts()
df = pd.DataFrame(data={'exams':tmp.values}, index=tmp.index).reset_index()
print(df)


# In[ ]:


plot = sns.barplot(x='sex', y='exams', hue='target', data=df)
plt.show()


# In[ ]:


# target vs age_approx
tmp = train_df.groupby('age_approx')['target'].value_counts()
df = pd.DataFrame(data={'exams':tmp.values}, index=tmp.index).reset_index()


# In[ ]:


plot = sns.barplot(x='age_approx', y='exams', hue='target', data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# target vs anatom_site_general_challenge
tmp = train_df.groupby('anatom_site_general_challenge')['target'].value_counts()
df = pd.DataFrame(data={'exams':tmp.values}, index=tmp.index).reset_index()
print(df)


# In[ ]:


plot = sns.barplot(x='anatom_site_general_challenge', y='exams', hue='target', data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# target vs anatom_site_general_challenge
tmp = train_df.groupby('diagnosis')['target'].value_counts()
df = pd.DataFrame(data={'exams':tmp.values}, index=tmp.index).reset_index()
print(df)


# In[ ]:


# Unique patient ids in train and test
print(f"Out of {len(train_df)} samples, only {len(train_df.patient_id.unique())} patient ids are unique in train set.")
print(f"Out of {len(test_df)} samples, only {len(test_df.patient_id.unique())} patient ids are unique in test set.")


# # Checking patient overlap in train and test

# In[ ]:


unique_patient_ids_train = set(train_df.patient_id.unique())
unique_patient_ids_test = set(test_df.patient_id.unique())
common_patient_ids = unique_patient_ids_train.intersection(unique_patient_ids_test)
print(f"There are totally {len(common_patient_ids)} common patient ids in train and test")


# There are no overlaps between train and test! Let's check how samples are distributed across patients.

# In[ ]:


#checking distribution of images for patients
sns.countplot(train_df.patient_id)


# It can be seen that few patients have very large number of image samples upto 115, but almost all the patients have multiple image samples.

# In[ ]:


print(f"{sum(train_df['target'])} tumour samples are contributed by {len(train_df.loc[train_df.target==1]['patient_id'].unique())} unique patients")


# # Sneak Peek into malignant samples which were taken from same patient!

# In[ ]:


tmp = train_df.groupby('patient_id')['target'].value_counts()
print(tmp)


# In[ ]:


df = pd.DataFrame(data={'exams':tmp.values}, index=tmp.index).reset_index()
print(df)


# In[ ]:


multiple_samples_df = df.query('target == 1 & exams > 1')[['patient_id','exams']]
multiple_samples_df


# Let's take a particular patient_id: IP_9997715 which has 3 tumour samples. Let's examine if 3 samples are diagnosed in different sections of body or they're just duplicates!!

# In[ ]:


train_df.query('patient_id == "IP_9997715" & target == 1')


# It's clear that those samples belong to same woman taken either at different sections at same age or at same age at different sections. Let's examine with another patient.

# In[ ]:


train_df.query('patient_id == "IP_9111321" & target == 1')


# Oops! It looks like this male patient's samples were taken at same age and at same section of the body. It's time to examine the image samples of this patient.

# In[ ]:


images = list(train_df.query('patient_id == "IP_9111321" & target == 1')['image_name'])
print(images)


# In[ ]:


images[0]


# In[ ]:


plt.figure(figsize=(15,10))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    img = plt.imread(os.path.join(img_dir, images[i]+'.jpg'))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(images[i])


# Samples are all different, though taken from same patient at same age and at same section of the body.
