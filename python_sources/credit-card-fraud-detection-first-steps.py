#!/usr/bin/env python
# coding: utf-8

# **This is the first step of analysis of dataset with fraud detection. I will upload new lines of code everyday to explore this dataset. **

# In[ ]:


from __future__ import division
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **1. Loading the dataset**

# In[ ]:


data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.name = 'Credit Card Fraud detection'

print('Number of features: %s' %data.shape[1])
print('Number of examples: %s' %data.shape[0])


# In[ ]:


data.head()


# **2. Analysing null values**

# In[ ]:


for col in data.columns:
    if data[col].isnull().sum() is not None:
        print('No null values in column: %s' %col)


# **3. Checking info, shape and any other information of dataset** 

# In[ ]:


def info_shape_data(dataset, dataset_name):
    print('Information about dataset: %s' %dataset_name)
    dataset.info()
    print('\n')
    print('Shape of dataset: %s' %dataset_name)
    print(dataset.shape)


# In[ ]:


info_shape_data(data, 'Credit Card Fraud Detection')


# In[ ]:


data.describe()


# **4. Target column - checking distribution**

# In[ ]:


fig, ax = plt.subplots(figsize=(17,8))
plt.title('Distribution of target value', fontsize=15)
plt.xlabel('Target', fontsize=13)
plt.ylabel('Count', fontsize=13)
vis1 = sns.countplot(data.Class, palette='GnBu_d')

for p in vis1.patches:
    vis1.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=13)


# In[ ]:


def target_class_distribution(dataset, target_column):
    target_sum = len(dataset[target_column])
    detected_target = len(dataset[target_column][dataset[target_column] == 1])
    detected_percent = round(detected_target/target_sum *100,3)
    not_detected_target = len(dataset[target_column][dataset[target_column] == 0])
    non_detected_percent = round(not_detected_target/target_sum*100,3)

    print('Detected frauds: %s' %detected_target)
    print('Non-detected frauds: %s' %not_detected_target)

    print('Percentage of detected frauds: %s' %detected_percent)
    print('Percentage of not-detected frauds: %s' %non_detected_percent)


# In[ ]:


target_class_distribution(data, 'Class')

