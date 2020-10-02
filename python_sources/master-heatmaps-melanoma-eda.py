#!/usr/bin/env python
# coding: utf-8

# # Melanoma
# 
# Melanoma, also known as malignant melanoma, is a type of skin cancer that develops from the pigment-producing cells known as melanocytes. It is the most common form of cancer found in US. It is the deadliest form of cancer as well. 
# 
# Nearly ***90%*** of all cases are caused by exposure to UV and sunlight.

# # Introduction
# 
# We have to identify melanoma in images from skin lesions. We'll use images within the same patient and determine which are likely to represent a melanoma. But before tmoving on to models we'll explore.
# 
# This notebook is just a start. Stay tuned for more !!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import math
from matplotlib import pyplot as plt
import seaborn as sns

# We don't need much libraries for now, we'll keep adding them as we move on.


# In[ ]:


main_path = '/kaggle/input/siim-isic-melanoma-classification/'
sub = pd.read_csv(main_path + '/sample_submission.csv')
te = pd.read_csv(main_path + '/test.csv')
tr = pd.read_csv(main_path + '/train.csv')


# In[ ]:


tr.head(5) #taking a look at data


# Most of the attributes in the train dataset are present. Very few are missing, we'll deal with them later.

# In[ ]:


tr.info()


# # Is Age Just a Number?
# 
# * In both, males and females we can see a bell curve at around 45 & 50 respectively. 
# * So does this mean that most of middle aged people are most affected by melanoma? We'll dig deeper next.

# In[ ]:


f = 1.4


# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['age_approx']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Sex', fontsize=20)


# # Two ways to look at a heatmap
# **We need to look at this vertically.** 
# * Although the highest number of cases were present for Age group(45 -50), we can clearly see they mostly have benign cases.
# * We can see, the malignant cases are low in younger age group but slowly increase towards the end with almost 20% of cases being malignant for 90 year olds.

# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['age_approx'],tr['benign_malignant']).apply(lambda r: r/r.sum()*100, axis=1)).T.round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Harmful or not?', fontsize=20)


# **Look horizontally.** 
# * In previous chart we saw, that most of cases for 90 year olds, were malignant. However, they comprise of a very small number in total **malignant** cases.
# * Hence, here on moving horizontally we can see that most malignant cases are present in the 60-65 age range.

# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['age_approx'],tr['benign_malignant']).T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Harmful or not?', fontsize=20)


# # Which Body Parts are mostly captured? 
# 
# * It is Torso and Lower extremity universally however in varying proportions.

# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['anatom_site_general_challenge']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Body Parts', fontsize=20)
plt.ylabel('Sex', fontsize=20)


# * For almost all age groups we can find that more than 50% cases have images of Torso.
# * Lower & Upper Extremity follow closely after covering the next 40-45 % cases.

# In[ ]:


sns.set(rc={'figure.figsize':(22,15)})
m = (pd.crosstab(tr['age_approx'],tr['anatom_site_general_challenge']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Body Parts', fontsize=20)
plt.ylabel('Age', fontsize=20)


# *** NOTE: I have taken all fields uptil here, but the unknown field in diagnosis appears to eat away the heatmap. From now on, I have removed the 'Unknown' field from Diagnosis.**

# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['diagnosis']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Sex', fontsize=20)


# # What is the diagnosis?
# 
# * **NOTE: I have removed 'Unknown' category from Diagnosis as of now.**
# 
# * We can see there are high number of cases for nevus. A melanocytic nevus is a type of melanocytic tumor consists of nevus-cell and commonly known as a mole. More can be read [here](https://en.wikipedia.org/wiki/Melanocytic_nevus).

# In[ ]:


sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['diagnosis'],tr['sex'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Sex', fontsize=20)


# # Diagnosis vs Age
# 
# * As I have removed the 'Unknown' column from the Diagnosis. We can see the top 2 age tiers are greyed out as they contained values from 'Unknown' column.
# * Melanoma has high detection for age group(15 - 20), but that would mostly be due to less number of cases available. Or is it children are more susceptible to melanoma?
# * However, there is an ever increasing trend from age 50 onwards which surely solidifies our older heatmaps that older people are highly susceptible of malignant tumuors. 
# * Higher the chance of Melanoma, higher is chance of malignancy.

# In[ ]:


sns.set(rc={'figure.figsize':(22,15)})
m = (pd.crosstab(tr['diagnosis'],tr['age_approx'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Age', fontsize=20)


# # Are some Body Parts more prone to Melanoma?
# 
# * Almost **every 4 of 5** lesions found on palms, soles, oral or genital areas  is a melanoma.

# In[ ]:


sns.set(rc={'figure.figsize':(16,10)})
m = (pd.crosstab(tr['diagnosis'],tr['anatom_site_general_challenge'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Body Parts', fontsize=20)


# **This is a notebook in progress....
# If you feel this was helpful, you know what to do ;)**
# 
# **Thank you.**
