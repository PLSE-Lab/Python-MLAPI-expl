#!/usr/bin/env python
# coding: utf-8

# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading the dataset

# In[ ]:


df=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# # Analyzing the data for null values and dropping the rows having an empty value

# In[ ]:


df.head(7)


# In[ ]:


df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# # We will predict wether the tumor is Malignant or Benign on the basis of radius,texture,smoothness,compactness, and concavity

# In[ ]:


x=df[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean"]]


# In[ ]:


y=df["diagnosis"]


# # Splitting the data for training and testing

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# # Using the model of LogisticRegression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# # Training the model and making predictions

# In[ ]:


logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)


# In[ ]:


df1=x_test


# In[ ]:


len(x_test)


# In[ ]:


df1


# In[ ]:


predictions


# In[ ]:





# In[ ]:





# # Checking for the accuracy score using jaccard_similarity_score

# In[ ]:


from sklearn.metrics import jaccard_similarity_score


# In[ ]:


accuracy_score=jaccard_similarity_score(y_test,predictions)
print(accuracy_score*100)


# # We have got an accuracy score of 89.36% which is an awesome score.

# In[ ]:


from sklearn.metrics import confusion_matrix


# ## Giving a look to the confusion matrix

# In[ ]:


matrix=confusion_matrix(y_test,predictions)
print(matrix)


# # Now, we will be using SVM as our second model and then we will compare the accuracy with the KNN model.

# In[ ]:


from sklearn import svm


# In[ ]:


clf=svm.SVC(gamma="scale")


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


predictions=clf.predict(x_test)


# In[ ]:


accuracy_score=jaccard_similarity_score(y_test,predictions)
print(accuracy_score*100)


# # We have got an accuracy score of 86.17% which is an awesome score.

# # We have got an accuracy score of 89.36% with the KNN model and 86.17% with the SVM model.

# In[ ]:




