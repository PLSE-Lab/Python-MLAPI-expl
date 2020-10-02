#!/usr/bin/env python
# coding: utf-8

# **** Indian Liver Patient Records
# Patient records collected from North East of Andhra Pradesh, India****

# **Context**
# 
# Patients with Liver disease have been continuously increasing because of excessive consumption of alcohol, inhale of harmful gases, intake of contaminated food, pickles and drugs. This dataset was used to evaluate prediction algorithms in an effort to reduce burden on doctors.
# 
# **Content**
# 
# This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.
# 
# Any patient whose age exceeded 89 is listed as being of age "90".

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data=pd.DataFrame(pd.read_csv('../input/indian_liver_patient.csv'))


# In[3]:


data.head()


# In[4]:


data.info()


# I tried to understand the attributes
# 
# **Total Bilirubin (Blood) / Total serum bilirubin, TSB **
# 
# * Measure of total amount of bilirubin.
# * It is often part of a panel of tests that measure liver function. 
# * A small amount of bilirubin in your blood is normal, but a high level may be a sign of liver disease.
# * Test results may vary depending on your age, gender, health history, the method used for the test, and other things.
# * Adults with jaundice generally have bilirubin levels greater than 2.5 milligrams per deciliter (mg/dL). 
# 
# **Direct Bilirubin/Conjugated bilirubin**
# 
# * Bilirubin is also part of bile, which your liver makes to help digest the food you eat.
# * Bilirubin that is bound to a certain protein (albumin) in the blood is called unconjugated, or indirect, bilirubin. 
# * Conjugated, or direct, bilirubin travels from the liver into the small intestine. 
# * A very small amount passes into your kidneys and is excreted in your urine. 
# * This bilirubin also gives urine its distinctive yellow color.
# * High level indicates liver problems, such as hepatitis, or blockages, such as gallstones.
# 
# **An alkaline phosphatase (ALP)**
# * Measure of the amount of ALP in your blood. 
# * ALP mostly found in the liver, bones, kidneys, and digestive system. 
# * When the liver is damaged, ALP may leak into the bloodstream. 
# * High levels of ALP can indicate liver disease or bone disorders.
# 
# **Aspartate Aminotransferase (AST) **
# 
# * Normal ranges are:
# * Males: 10 to 40 units/L
# * Females: 9 to 32 units/L
# 
# **Alamine Aminotransferase**
# 
# * A normal ALT test result can range from 7 to 55 units per liter (U/L). 
# * Levels are normally higher in men.
# 
# **Total Protein**
# 
# * The normal range for total protein is between 6 and 8.3 grams per deciliter (g/dL). 
# 
# **Albumin**
# 
# * A normal albumin range is 3.4 to 5.4 g/dL. 
# * grams per deciliter (g/dL).
# * If you have a lower albumin level,It can mean that you have liver disease or an inflammatory disease.
# 
# **Albumin and Globulin Ratio**
# * A normal range of albumin is 39 to 51 grams per liter (g/L) of blood. 
# * The normal range for globulins varies by specific type. A normal range for total globulins is 23 to 35 g/L. 
# * If your protein level is low, you may have a liver or kidney problem.
# 

# **Dataset Feature**
# 
# * Value 1  -->  Group 1
# * Value 2  -->  Group 2

# In[17]:


data.sample(5)


# In[6]:


data.shape


# * There is no metrics information about the features

# In[7]:


data.describe()


# There are 4 missing values in Albumin_and_Globulin_Ratio 

# In[8]:


missing_values = data.isnull().sum()
missing_values


# Lets drop those missing values

# In[9]:


data=data.dropna()
data.shape


# * Create new categorical column Health 

# In[10]:


print('Number of people in Dataset 1',data[data['Dataset'] == 1].Age.count())
print('Number of people in Dataset 2',data[data['Dataset'] == 2].Age.count())


# **Basic EDA on Labels**

# Number of people in Dataset 1 and 2

# In[11]:


sns.countplot(x='Dataset',data=data)
plt.show()


# > Display Datasets by patient gender

# In[12]:


# Gender Distribution of 2 Dataset
sns.countplot(x='Gender',data=data,hue='Dataset',palette="Set1")
plt.title('Distribution of Datasets by Gender')
plt.show()


# Okay! There are many males in the dataset 1

# **Distribution of features in the whole dataset**

# In[22]:


columns=list(data.columns[:10])
columns.remove('Gender')
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    data[i].hist(bins=10,edgecolor='black')#,range=(0,0.3))
    plt.title(i)
plt.show()


# *  Check age distribution by sex

# In[14]:


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(wspace=0.2,hspace=0.5)
data[data['Gender']=='Male'].Age.plot(ax=ax1, kind='hist', bins=10,edgecolor='black')
ax1.set_title('Male Distribution')
data[data['Gender']=='Female'].Age.plot(ax=ax2, kind='hist',bins=10,edgecolor='black')
ax2.set_title('Female Distribution')
plt.show()


# **Distribution of Age by Gender in Dataset 1**

# In[15]:


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(wspace=0.2,hspace=0.5)
data[(data['Gender']=='Male') & (data['Dataset'] == 1)].Age.plot(ax=ax1, kind='hist', bins=10,edgecolor='black')
ax1.set_title('Male Distribution')
data[(data['Gender']=='Female') & (data['Dataset'] == 1)].Age.plot(ax=ax2, kind='hist',bins=10,edgecolor='black')
ax2.set_title('Female Distribution')
plt.show()


# **Distribution of Age by Gender in Dataset 2**

# In[16]:


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(wspace=0.2,hspace=0.5)
data[(data['Gender']=='Male') & (data['Dataset'] == 2)].Age.plot(subplots=True,ax=ax1, kind='hist', bins=10,edgecolor='black')
ax1.set_title('Male Distribution')
data[(data['Gender']=='Female') & (data['Dataset'] == 2)].Age.plot(subplots=True,ax=ax2, kind='hist',bins=10,edgecolor='black')
ax2.set_title('Female Distribution')
plt.show()


# ********************************To Be Continued****
