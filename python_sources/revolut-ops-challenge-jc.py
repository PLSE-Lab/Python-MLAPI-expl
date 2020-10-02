#!/usr/bin/env python
# coding: utf-8

# In[188]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualisation
import statsmodels.formula.api as sm #running stats functions
from datetime import datetime #date and time convertor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[189]:


##Importing New Dataset## 
#Loads the dataset as a CSV file, "data"
data = pd.read_csv("../input/JC_Revolut Database CSV_Final.csv", low_memory = False)


# In[190]:


#Converting 'created_at' into datetime format; changes object to string
data['created_at'] =  pd.to_datetime(data['created_at'])


# In[191]:


##Definition of New Feature## 
#Defines Feature data as only results whose status is 'clear' or 'consider'
data = data[(data.result == 'clear') | (data.result == 'consider')] 


# In[192]:


#Checking the shape of the dataset to determine total number of KYC attempts
data.shape


# In[193]:


#Defines Feature docpasses as only doc results whose status is 'clear'
docpasses = data[(data.result == 'clear')]


# In[194]:


#Determine total number of successful document passes

docpasses.shape


# In[195]:


#Defines Feature docpasses as only face results whose status is 'clear'
facepasses = data[(data.result2 == 'clear')]


# In[196]:


#Determine total number of successful facial passes

facepasses.shape


# In[197]:


#Defines Feature kycpasses as attempts with both doc and face results as 'clear'
kycpasses = data[(data.result2 == 'clear') & (data.result == 'clear')] 


# In[198]:


#Determine total number of successful KYC passes

kycpasses.shape


# In[199]:


#Creating a copy of the main dataset for Unique IDs
data_unq=data.copy(deep=True)


# In[200]:


#Dropping duplicate User IDs but keeping the first instance, dataset 'data_unq'
data_unq.drop_duplicates(subset ="user_id", inplace = True)


# In[201]:


#Create a new column 'target' that gives 'True' if both conditons are met - i.e. KYC Passes
data_unq['target'] = (data_unq.result2 == 'clear') & (data_unq.result == 'clear')


# In[202]:


#Determining the total number of unique User IDs
data_unq.shape


# In[203]:


#Determining count of KYC fails and passes; False=fail, Pass=True
from collections import Counter

Counter(data_unq.target)


# In[204]:


#Creating new dataset for unique KYC fails
kycfails_unq=data_unq.copy (deep=True)


# In[205]:


#Filtering for unique KYC fails
kycfails_unq = kycfails_unq[(kycfails_unq.result2 == 'consider') | (kycfails_unq.result == 'consider')] 


# In[206]:


plt.hist(kycfails_unq.created_at,500)


# In[207]:


#Deep dive into Oct 2017

oct17=kycfails_unq[(kycfails_unq['created_at'] > '2017-10-01') & (kycfails_unq['created_at'] < '2017-10-31')]


# In[208]:


plt.hist(oct17.created_at,50)


# In[209]:



plt.hist(kycfails_unq.created_at, 50, alpha=0.5)
plt.hist(data_unq.created_at, 50, alpha=0.5)


# In[210]:


#Define attempts from June to October
JtO_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-06-01') & (kycfails_unq['created_at'] < '2017-10-31')]


# In[211]:


JtO = data_unq[(data_unq['created_at'] > '2017-06-01') & (data_unq['created_at'] < '2017-10-31')]


# In[212]:


len(JtO_failsunq)/len(JtO)*100


# In[213]:


jun17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-06-01') & (kycfails_unq['created_at'] < '2017-06-30')]


# In[214]:


jun17 = data_unq[(data_unq['created_at'] > '2017-06-01') & (data_unq['created_at'] < '2017-06-30')]


# In[215]:


len(jun17_failsunq)/len(jun17)*100


# In[216]:


aug17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-08-01') & (kycfails_unq['created_at'] < '2017-08-31')]


# In[217]:


aug17 = data_unq[(data_unq['created_at'] > '2017-08-01') & (data_unq['created_at'] < '2017-08-31')]


# In[218]:


len(aug17_failsunq)/len(aug17)*100


# In[219]:


oct17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-10-01') & (kycfails_unq['created_at'] < '2017-10-31')]


# In[220]:


oct17 = data_unq[(data_unq['created_at'] > '2017-10-01') & (data_unq['created_at'] < '2017-10-31')]


# In[221]:


len(oct17_failsunq)/len(oct17)*100


# In[222]:


Counter(data_unq.image_quality_result)


# In[223]:


Counter(data_unq.police_record_result)


# In[224]:


Counter(data_unq.compromised_document_result)


# In[225]:


Counter(data_unq.facial_image_integrity_result)


# In[226]:


Counter(data_unq.face_comparison_result)


# In[227]:


Counter(data_unq.facial_visual_authenticity_result)


# In[228]:


#Creating new dataset for when image quality = clear in KYC fails
kycfails_unq_imgclear=kycfails_unq.copy (deep=True)


# In[229]:


kycfails_unq_imgclear=kycfails_unq_imgclear[(kycfails_unq_imgclear.image_quality_result == 'clear')]


# In[230]:


len(kycfails_unq_imgclear)


# In[231]:


Counter(kycfails_unq_imgclear.face_detection_result)


# In[232]:


Counter(kycfails_unq_imgclear.colour_picture_result)


# In[233]:


Counter(kycfails_unq_imgclear.visual_authenticity_result)


# In[234]:


Counter(kycfails_unq_imgclear.data_validation_result)


# In[235]:


Counter(kycfails_unq_imgclear.data_consistency_result)


# In[239]:


#Creating new dataset for when image quality = unidentified or facial image intergrity = consider in KYC fails
kycfails_unq_imgp=kycfails_unq.copy (deep=True)


# In[240]:



kycfails_unq_imgp=kycfails_unq_imgp[(kycfails_unq_imgp.image_quality_result=='unidentified')|(kycfails_unq_imgp.facial_image_integrity_result=='consider')]


# In[241]:


plt.hist(kycfails_unq.created_at, 50, alpha=0.5)
plt.hist(kycfails_unq_imgp.created_at, 50, alpha=0.5)

