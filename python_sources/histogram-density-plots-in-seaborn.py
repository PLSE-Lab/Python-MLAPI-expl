#!/usr/bin/env python
# coding: utf-8

# ## Histogram and Density plots to understand pattern in Breast Cancer Tumor
# Our dataset containing information collected from microscopic images of breast cancer tumors, similar to the image below.
# 
# ![ex4_cancer_image](https://i.imgur.com/qUESsJe.png)
# 
# Each tumor has been labeled as either [**benign**](https://en.wikipedia.org/wiki/Benign_tumor) (_noncancerous_) or **malignant** (_cancerous_).
# 
# A **tumor** is an abnormal lump or growth of cells. When the cells in the tumor are **normal**, it is **benign**. Something just went wrong, and they overgrew and produced a lump. When the cells are **abnormal** and can grow uncontrollably, they are cancerous cells, and the tumor is **malignant**.
# 
# To learn more about how this kind of data is used to create intelligent algorithms to classify tumors in medical settings, **watch the short video [at this link](https://www.youtube.com/watch?v=9Mz84cwVmS0)**!

# ## Dataset
# We have two dataset:  
# 1.Cancer Benign Data(Noncancerous)  
# 2.Cancer Maligant Data(Cancerous)
# 

# ## Import statements

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Read Data

# In[ ]:


#Read the Benign data
cancer_b_data = pd.read_csv('../input/cancer_b.csv',index_col = 'Id')

#review the benign data
cancer_b_data.head()


# In[ ]:


#Read the maligant data
cancer_m_data = pd.read_csv('../input/cancer_m.csv',index_col = 'Id')

#review mailgant data
cancer_m_data.head()


# # Histogram
# 
# Plot for Area(mean) for both Benign and Maligant Tumor

# In[ ]:


# Histograms for benign tumor
sns.distplot(cancer_b_data['Area (mean)'],label = "b_data",kde = True) # Your code here (benign tumors)

#Histogram for maligant tumor
sns.distplot(cancer_m_data['Area (mean)'],label = "m_data",kde = True) # Your code here (malignant tumors)

#title 
plt.title("Analyzing Area(mean) of both Benign and Maligant tumors")

#for labels
plt.legend()


# Histogram Plot for Radius(mean) for both Benign and Maligant Tumor

# In[ ]:


# Histograms for benign tumor
sns.distplot(cancer_b_data['Radius (mean)'],label = "b_data",kde = True) # Your code here (benign tumors)

#Histogram for maligant tumor
sns.distplot(cancer_m_data['Radius (mean)'],label = "m_data",kde = True) # Your code here (malignant tumors)

#title 
plt.title("Analyzing Radius(mean) of both Benign and Maligant tumors")

#for labels
plt.legend()


# Histogram Plot for Perimeter(mean) for both Benign and Maligant Tumor

# In[ ]:


# Histograms for benign tumor
sns.distplot(cancer_b_data['Perimeter (mean)'],label = "b_data",kde = True) # Your code here (benign tumors)

#Histogram for maligant tumor
sns.distplot(cancer_m_data['Perimeter (mean)'],label = "m_data",kde = True) # Your code here (malignant tumors)

#title 
plt.title("Analyzing Perimeter(mean) of both Benign and Maligant tumors")

#for labels
plt.legend()


# # KDE Plots
# KDE Plot for Area(mean) for both Benign and Maligant Tumor
# 

# In[ ]:


# KDE plot for Benign
sns.kdeplot(cancer_b_data['Area (mean)'], label="b_data", shade=True)

#KDE pot for Maligant
sns.kdeplot(cancer_m_data['Area (mean)'], label="m_data", shade=True)

#Title
plt.title("Analyzing Area(mean) of both benign and maligant data ")

plt.legend()


# KDE Plot for Radius(mean) for both Benign and Maligant Tumor

# In[ ]:


# KDE plot for Benign
sns.kdeplot(cancer_b_data['Radius (mean)'], label="b_data", shade=True)

#KDE pot for Maligant
sns.kdeplot(cancer_m_data['Radius (mean)'], label="m_data", shade=True)

#Title
plt.title("Analyzing Radius (mean) of both benign and maligant data ")

plt.legend()


# KDE Plot for Perimeter(mean) for both Benign and Maligant Tumor

# In[ ]:


# KDE plot for Benign
sns.kdeplot(cancer_b_data['Perimeter (mean)'], label="b_data", shade=True)

#KDE pot for Maligant
sns.kdeplot(cancer_m_data['Perimeter (mean)'], label="m_data", shade=True)

#Title
plt.title("Analyzing Perimeter (mean) of both benign and maligant data ")

plt.legend()


# In[ ]:




