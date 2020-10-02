#!/usr/bin/env python
# coding: utf-8

# Ref: https://www.kaggle.com/akshat0007/cardiotocology
# 
# https://pycaret.org/transformation/
# 
# https://www.stat.umn.edu/arc/yjpower.pdf
# 
# 

# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


#import
import pandas as pd
import missingno
from pycaret.classification import *
import numpy as np


# In[ ]:


df = pd.read_csv("../input/fetalhr/CTG.csv")
df.head()


# FileName: of CTG examination
# 
# Date: of the examination
# 
# b: start instant
# 
# e: end instant
# 
# LBE: baseline value (medical expert)
# 
# LB: baseline value (SisPorto)
# 
# AC: accelerations (SisPorto)
# 
# FM: foetal movement (SisPorto)
# 
# UC: uterine contractions (SisPorto)
# 
# ASTV: percentage of time with abnormal short term variability (SisPorto)
# 
# mSTV: mean value of short term variability (SisPorto)
# 
# ALTV: percentage of time with abnormal long term variability (SisPorto)
# 
# mLTV: mean value of long term variability (SisPorto)
# 
# DL: light decelerations
# 
# DS: severe decelerations
# 
# DP: prolongued decelerations
# 
# DR: repetitive decelerations
# 
# Width: histogram width
# 
# Min: low freq. of the histogram
# 
# Max: high freq. of the histogram
# 
# Nmax: number of histogram peaks
# 
# Nzeros: number of histogram zeros
# 
# Mode: histogram mode
# 
# Mean: histogram mean
# 
# Median: histogram median
# 
# Variance: histogram variance
# 
# Tendency: histogram tendency: -1=left assymetric; 0=symmetric; 1=right assymetric
# 
# A: calm sleep
# 
# B: REM sleep
# 
# C: calm vigilance
# 
# D: active vigilance
# 
# SH: shift pattern (A or Susp with shifts)
# 
# AD: accelerative/decelerative pattern (stress situation)
# 
# DE: decelerative pattern (vagal stimulation)
# 
# LD: largely decelerative pattern
# 
# FS: flat-sinusoidal pattern (pathological state)
# 
# SUSP: suspect pattern
# 
# CLASS: Class code (1 to 10) for classes A to SUSP
# 
# NSP: Normal=1; Suspect=2; Pathologic=3
# 
# The dataset publisher used these as feature and decision columns(I keep it for hint)
# 
# X=df[['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',
#        'DS', 'DP', 'DR']]
# Y=df[["NSP"]]

# # Missing values

# In[ ]:


#plot graphic of missing value
missingno.matrix(df, figsize = (30, 10))


# Decision : I think, I can go away with this data
# but let's check if there is any?

# In[ ]:


df.isnull().sum()
#there are some actually


# Using PyCaret for preparing my data

# In[ ]:


my_data = setup(data = df, 
                numeric_imputation = "median", 
                categorical_imputation = "mode", 
                target = "NSP", train_size = 0.8, 
                transformation = True, 
                transformation_method = "yeo-johnson",
               )


# In[ ]:


#Trying to Understand my_data
print(type(my_data))
print(np.shape(my_data))
print(my_data[1])
#I don't understand much, help me in the comments


# # Let's Predict

# In[ ]:




