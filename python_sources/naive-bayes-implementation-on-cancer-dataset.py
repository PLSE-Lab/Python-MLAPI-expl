#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Algorithm Implementation on Cancer Dataset

# ## Content:
# 
# 1. [Importing Dataset](#1)
# 1. [Getting Info About Dataset](#2)
# 1. [Dataset Visualization](#3)
# 1. [Meaning Of Naive Bayes Algorithm](#4)
# 1. [Naive Bayes with Sklearn](#5)

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


# <a id = "1"></a>
# 
# ## 1. Importing Dataset:

# In[ ]:


dataset = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# > <a id = "2"></a>
# # 2. Getting Info About Dataset

# Dataset information:
# 
# * Dataset Characteristics: Multivariate
# * Attribute Characteristics: Real
# * Attribute Characteristics: Classification
# * Number of Instances: 569
# * Number of Attributes: 32
# * Missing Values: No

# Column names and meanings:
# * id: ID number
# * diagnosis: The diagnosis of breast tissues (M = malignant, B = benign)
# * radius_mean: mean of distances from center to points on the perimeter
# * texture_mean: standard deviation of gray-scale values
# * perimeter_mean: mean size of the core tumor
# * area_mean: area of the tumor
# * smoothness_mean: mean of local variation in radius lengths
# * compactness_mean: mean of perimeter^2 / area - 1.0
# * concavity_mean: mean of severity of concave portions of the contour
# * concave_points_mean: mean for number of concave portions of the contour
# * symmetry_mean
# * fractal_dimension_mean: mean for "coastline approximation" - 1
# * radius_se: standard error for the mean of distances from center to points on the perimeter
# * texture_se: standard error for standard deviation of gray-scale values
# * perimeter_se
# * area_se
# * smoothness_se: standard error for local variation in radius lengths
# * compactness_se: standard error for perimeter^2 / area - 1.0
# * concavity_se: standard error for severity of concave portions of the contour
# * concave_points_se: standard error for number of concave portions of the contour
# * symmetry_se
# * fractal_dimension_se: standard error for "coastline approximation" - 1
# * radius_worst: "worst" or largest mean value for mean of distances from center to points on the perimeter
# * texture_worst: "worst" or largest mean value for standard deviation of gray-scale values
# * perimeter_worst
# * area_worst
# * smoothness_worst: "worst" or largest mean value for local variation in radius lengths
# * compactness_worst: "worst" or largest mean value for perimeter^2 / area - 1.0
# * concavity_worst: "worst" or largest mean value for severity of concave portions of the contour
# * concave_points_worst: "worst" or largest mean value for number of concave portions of the contour
# * symmetry_worst
# * fractal_dimension_worst: "worst" or largest mean value for "coastline approximation" - 1

# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# Now, let's get rid of "id" and "Unnamed: 32" features because we don't need to use them while diagnosing whether the patient has a cancer or not. 

# In[ ]:


dataset = dataset.drop(["id"], axis = 1)


# In[ ]:


dataset = dataset.drop(["Unnamed: 32"], axis = 1)


# In[ ]:


dataset.head(3)


# In[ ]:


M = dataset[dataset.diagnosis == "M"]


# In[ ]:


M.head(5)


# In[ ]:


B = dataset[dataset.diagnosis == "B"]


# In[ ]:


B.head(5)


# <a id = "3"></a>
# # 3. Dataset Visualization 

# In[ ]:


plt.title("Malignant vs Benign Tumor")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "lime", label = "Benign", alpha = 0.3)
plt.legend()
plt.show()


# <a id = "4"></a>
# # 4. Meaning Of Naive Bayes Algorithm

# ![](https://www.intelkit.com/wp-content/uploads/2020/04/Naive-bayes-1.png)

# <a id = "5"></a>
# # 5. KNN with Sklearn

# In[ ]:


dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]


# In[ ]:


x = dataset.drop(["diagnosis"], axis = 1)
y = dataset.diagnosis.values


# In[ ]:


# Normalization:
x = (x - np.min(x)) / (np.max(x) - np.min(x))


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)


# In[ ]:


print("Naive Bayes score: ",nb.score(x_test, y_test))

