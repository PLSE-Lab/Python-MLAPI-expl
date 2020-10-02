#!/usr/bin/env python
# coding: utf-8

# # Data_Dictionary
# 
# Elevation = Elevation in meters.
# 
# Aspect = Aspect in degrees azimuth.
# 
# Slope = Slope in degrees.
# 
# Horizontal_Distance_To_Hydrology = Horizontal distance to nearest surface water features.
# 
# Vertical_Distance_To_Hydrology = Vertical distance to nearest surface water features.
# 
# Horizontal_Distance_To_Roadways = Horizontal distance to nearest roadway.
# 
# Hillshade_9am = Hill shade index at 9am, summer solstice. Value out of 255.
# 
# Hillshade_Noon = Hill shade index at noon, summer solstice. Value out of 255.
# 
# Hillshade_3pm = Hill shade index at 3pm, summer solstice. Value out of 255.
# 
# Horizontal_Distance_To_Fire_Point = sHorizontal distance to nearest wildfire ignition points.
# 
# Wilderness_Area1 = Rawah Wilderness Area
# 
# Wilderness_Area2 = Neota Wilderness Area
# 
# Wilderness_Area3 = Comanche Peak Wilderness Area
# 
# Wilderness_Area4 = Cache la Poudre Wilderness Area
# 
# Soil_Type1 to Soil_Type40 [Total 40 Types]
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})


# In[ ]:


data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


# removing the id column
X = data.drop(columns='Id')


# # **The aim of this excercise is to find correlations between different characteristics of forest such as elevation,slope,soil type, etc to the Cover_Type of a forest. Thus, we will use EDA to better understand the data**

# # **we will classify the characteristics into subgroups to better understand the data**

# # **Y1 = 'Elevation', 'Aspect', 'Slope'(group based on the geographical characteristics)**

# In[ ]:


Y1 = X.loc[:, 'Elevation':'Slope' ]


# In[ ]:


for i,col in enumerate(Y1):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])


# In[ ]:


Y1 = pd.concat([Y1, X['Cover_Type']], axis=1)


# In[ ]:


sns.pairplot(Y1, hue ='Cover_Type')


#  # **Y2 = 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'  **

# In[ ]:


Y2 = X[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]


# In[ ]:


for i,col in enumerate(Y2):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])


# In[ ]:


Y2 = pd.concat([Y2, X['Cover_Type']], axis=1)


# In[ ]:


sns.pairplot(Y2, hue ='Cover_Type')


# # **Y3 = 'Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm' **

# In[ ]:


Y3= X[['Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm']]


# In[ ]:


for i,col in enumerate(Y3):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])


# In[ ]:


Y3 = pd.concat([Y3, X['Cover_Type']], axis=1)


# In[ ]:


sns.pairplot(Y3, hue ='Cover_Type')


# # **Y4 = 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4' **

# In[ ]:


Y4 = X[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']]


# In[ ]:


for i,col in enumerate(Y4):
    plt.figure(i, figsize=(12,5))
    y1 = []
    x1 = []
    for x, y in enumerate(X.groupby(by='Cover_Type')[col].sum()):
        y1.append(y)
        x1.append(x + 1)
    plt.bar(x1,y1 )
    plt.title(col)


# # **Y5 = All the soil types **

# In[ ]:


Y5 = X[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',]]


# In[ ]:


for i,col in enumerate(Y5):
    plt.figure(i, figsize=(6,4))
    y1 = []
    x1 = []
    for x, y in enumerate(X.groupby(by='Cover_Type')[col].sum()):
        y1.append(y)
        x1.append(x + 1)
    plt.bar(x1,y1)
    plt.title(col)


# In[ ]:




