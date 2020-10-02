#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')  # Reading the document with the extension of .csv with the Pandas


# In[ ]:


data.info() # informations about the data


# #               ** Explanations about the Data**
# 1. age - age in years
# 2. sex - sex (1 = male; 0 = female)
# 3. cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol - serum cholestoral in mg/dl
# 6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# 7. restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest
# 11. slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
# 12. ca - number of major vessels (0-3) colored by flourosopy
# 13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 

# In[ ]:


data.corr() # correlations between data


# In[ ]:


# LINE PLOT
# kind=kind of plot, color=color, label=label of data, linewidth=width of line, alpha=opacity, grid=grid, linestyle=style of line
data.oldpeak.plot(kind = 'line', color = 'b', label = 'Oldpeak', linewidth = '1',alpha = 0.8, grid = True, linestyle = ':')
data.ca.plot(kind = 'line', color = 'r', label = 'Ca', linewidth = '1', alpha = 0.8, grid = True, linestyle = '-.')
plt.legend(loc='upper right') # legend = puts label in the plot
plt.xlabel('Oldpeak') # label = name of label
plt.ylabel('Ca')
plt.title('Oldpeak-Ca Line Plot') # title = title of plot


# **On most part of the graph, we can see that Oldpeak is directly proportional to Ca.**

# In[ ]:


# SCATTER PLOT
# x = Age, y = Chol
data.plot(kind = 'scatter', x = 'age', y = 'chol', alpha = 0.5, color = 'red')
plt.xlabel('Age') # label = name of label
plt.ylabel('Chol')
plt.title('Age-Chol Scatter Plot') # title --> title of plot


# **In this graph, we can see that cholesterol is effective between the ages of 50-65.**

# In[ ]:


# Histogram
# bins = number of bar in figure
# figsize = size of figure
data.thalach.plot(kind = 'hist', bins = 60, figsize = (10,10))


# **In this graph, we can understand that the maximum heart rate achieved has a high frequency in the range of 130-175.**
