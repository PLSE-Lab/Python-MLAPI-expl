#!/usr/bin/env python
# coding: utf-8

# # Visualization Heart Disease Data
# 

# <img style="float: left;" src="https://www.nation.co.ke/image/view/-/3399802/highRes/1448503/-/maxw/600/-/100jq6i/-/heapic.jpg" width="350px"/>

# ## <b>Contents</b>
# 
# 1. [Introduction ](#section1)
# 2. [The Data](#section2)
# 3. [The Visualization Data](#section3)<br>
#   3.1.[Correlation Map](#section4)<br>
#   3.2.[Line Plot](#section5)<br>
#   3.3.[Scatter Plot](#section6)<br>
#   3.4.[Histogram](#section7)<br>

# <a id='section1'></a>

#   # 1. Introduction
# 

# This study is for the explain relationship of the heart disease effect. 
# Use python and many different libraries for the visualizations data. This work is begin the data science  and ml(machine learning), ml is include data science and many visualizations technics. This platform (kaggle) is support the many different python libraries. For the libraries used; numpy, pandas, seaborn, matplotlib. Especially pandas library  is important for the data processing.

# 

# In[ ]:


Import the libraries:


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id='section2'></a>

# 
#   # 2. The Data
# 

# Next, load the data:

# In[ ]:


data = pd.read_csv("../input/heart.csv")


# 

# It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
# 
# - **age**: The person's age in years
# - **sex**: The person's sex (1 = male, 0 = female)
# - **cp:** The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# - **trestbps:** The person's resting blood pressure (mm Hg on admission to the hospital)
# - **chol:** The person's cholesterol measurement in mg/dl
# - **fbs:** The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) 
# - **restecg:** Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# - **thalach:** The person's maximum heart rate achieved
# - **exang:** Exercise induced angina (1 = yes; 0 = no)
# - **oldpeak:** ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more [here](https://litfl.com/st-segment-ecg-library/))
# - **slope:** the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# - **ca:** The number of major vessels (0-3)
# - **thal:** A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# - **target:** Heart disease (0 = no, 1 = yes)
# 
# 

# Let's take a look;

# In[ ]:


data.info()


# In[ ]:


# Data's Columns:
data.columns


# The data's first **10** lines:

# In[ ]:


data.head(10)


# <a id='section4'></a>
# <a id='section3'></a>

# # 3. Visualization Tools:
# 
# 
# ## Correlation Map
# Visualization with **correlation map**

# In[ ]:


# Correlation map
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# <a id='section5'></a>

# ## Line Plot
# Visualization with **line plot**,  relation of **thalasemia** and **cholesterol**

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity
# grid = grid, linestyle = sytle of line

data.thal.plot(kind = 'line', color = 'g', label = ' Thalassemia',linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')
data.oldpeak.plot(color = 'r',label = 'Cholesterol',linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Thalassemia-Cholesterol Plot')
plt.show()


# A blood disorder called 'Thalassemia' (3 = normal; 6 = fixed defect; 7 = reversable defect)

# <a id='section6'></a>

# ## Scatter Plot
# Visualization with **scatter plot**,  relation of **age** and **cholesterol**

# In[ ]:


# Scatter Plot
# x = Age, y = Cholesterol

data.plot(kind = 'scatter',x = 'age', y = 'chol',alpha = 0.5, color = 'red')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age-Cholesterol Plot')


# <a id='section7'></a>

# ## Histogram
# **Age** frequency

# In[ ]:


# Histogram
# Age frequency in data
# bins = number of bar in figure

data.age.plot(kind = 'hist', bins = 50, figsize = (12,12))
plt.show()


# Thanks for my instructor [**DATAITEAM** ](http://https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners)
# 
# 
# Data from:[Heart Disease](https://www.kaggle.com/ronitf/heart-disease-uci)
# 
