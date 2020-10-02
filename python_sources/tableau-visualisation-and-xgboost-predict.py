#!/usr/bin/env python
# coding: utf-8

# * <h1><b>Competition Description<h1><br>
#     <br>
# ![image.png](attachment:image.png)

# <p>Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.<p>

# 
# <h1>File descriptions<h1>
# 
# train.csv - the training set
# test.csv - the test set
# data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
# sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms
# 
# Data fields<br>
# <h1>Here's a brief version of what you'll find in the data description file.<br><h1>
# 
# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.<br>
# MSSubClass: The building class<br>
# MSZoning: The general zoning classification<br>
# LotFrontage: Linear feet of street connected to property<br>
# LotArea: Lot size in square feet<br>
# Street: Type of road access<br>
# Alley: Type of alley access<br>
# LotShape: General shape of property<br>
# LandContour: Flatness of the property<br>
# Utilities: Type of utilities available<br>
# LotConfig: Lot configuration<br>
# LandSlope: Slope of property<br>
# Neighborhood: Physical locations within Ames city limits<br>
# Condition1: Proximity to main road or railroad<br>
# Condition2: Proximity to main road or railroad (if a second is present)<br>
# BldgType: Type of dwelling<br>
# HouseStyle: Style of dwelling<br>
# OverallQual: Overall material and finish quality<br>
# OverallCond: Overall condition rating<br>
# YearBuilt: Original construction date<br>
# YearRemodAdd: Remodel date<br>
# RoofStyle: Type of roof<br>
# RoofMatl: Roof material<br>
# Exterior1st: Exterior covering on house<br>
# Exterior2nd: Exterior covering on house (if more than one material)<br>
# MasVnrType: Masonry veneer type<br>
# MasVnrArea: Masonry veneer area in square feet<br>
# ExterQual: Exterior material quality<br>
# ExterCond: Present condition of the material on the exterior<br>
# Foundation: Type of foundation<br>
# BsmtQual: Height of the basement<br>
# BsmtCond: General condition of the basement<br>
# BsmtExposure: Walkout or garden level basement walls<br>
# BsmtFinType1: Quality of basement finished area<br>
# BsmtFinSF1: Type 1 finished square feet<br>
# BsmtFinType2: Quality of second finished area (if present)<br>
# BsmtFinSF2: Type 2 finished square feet<br>
# BsmtUnfSF: Unfinished square feet of basement area<br>
# TotalBsmtSF: Total square feet of basement area<br>
# Heating: Type of heating<br>
# HeatingQC: Heating quality and condition<br>
# CentralAir: Central air conditioning<br>
# Electrical: Electrical system<br>
# 1stFlrSF: First Floor square feet<br>
# 2ndFlrSF: Second floor square feet<br>
# LowQualFinSF: Low quality finished square feet (all floors)<br>
# GrLivArea: Above grade (ground) living area square feet<br>
# BsmtFullBath: Basement full bathrooms<br>
# BsmtHalfBath: Basement half bathrooms<br>
# FullBath: Full bathrooms above grade<br>
# HalfBath: Half baths above grade<br>
# Bedroom: Number of bedrooms above basement level<br>
# Kitchen: Number of kitchens<br>
# KitchenQual: Kitchen quality<br>
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)<br>
# Functional: Home functionality rating<br>
# Fireplaces: Number of fireplaces<br>
# FireplaceQu: Fireplace quality<br>
# GarageType: Garage location<br>
# GarageYrBlt: Year garage was built<br>
# GarageFinish: Interior finish of the garage<br>
# GarageCars: Size of garage in car capacity<br>
# GarageArea: Size of garage in square feet<br>
# GarageQual: Garage quality<br>
# GarageCond: Garage condition<br>
# PavedDrive: Paved driveway<br>
# WoodDeckSF: Wood deck area in square feet<br>
# OpenPorchSF: Open porch area in square feet<br>
# EnclosedPorch: Enclosed porch area in square feet<br>
# 3SsnPorch: Three season porch area in square feet<br>
# ScreenPorch: Screen porch area in square feet<br>
# PoolArea: Pool area in square feet<br>
# PoolQC: Pool quality<br>
# Fence: Fence quality<br>
# MiscFeature: Miscellaneous feature not covered in other categories<br><br>
# MiscVal: $Value of miscellaneous feature <br>
# MoSold: Month Sold<br>
# YrSold: Year Sold<br>
# SaleType: Type of sale<br>
# SaleCondition: Condition of sale<br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1> Importing of Classes <h1>

# In[ ]:


#Importing python libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model


# In[ ]:


pd.set_option('display.max_rows',100)# amount of rows that can be seen at a time


# <h1>We are making  test and train  dataframe to analyse the given dataset.<h1>

# In[ ]:


# import train and test to play with it
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# <h1>lets check out our data to get info about it.

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


test_df['SalePrice'] = 0
# Adding a row in test data for further calcualtion


# In[ ]:


train_df.head()


# In[ ]:


# We are concating to get to know more about the complete dataset
df = pd.concat((train_df,test_df),axis = 0)
df = df.reset_index()


# In[ ]:


df.info()


# In[ ]:


df = df.drop(['index'],axis = 1)
df.tail()


# In[ ]:


df = df.set_index(['Id'])


# In[ ]:


df.head()


# <h2> After going through the dataset once , we know that we have to preprocess the complete data before applying machine learning algorithm <h2>

# In[ ]:


df.describe(include = 'all')


# In[ ]:


df.isnull().sum()


# In[ ]:


df.MSZoning.value_counts()


# In[ ]:


df.LotFrontage.value_counts()


# In[ ]:


df.to_csv('out.csv')


# In[ ]:




