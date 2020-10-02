#!/usr/bin/env python
# coding: utf-8

# Content
# The data consists of 10,000 observations of space taken by the SDSS. Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.
# 
# Feature Description
# The table results from a query which joins two tables (actuaclly views): "PhotoObj" which contains photometric data and "SpecObj" which contains spectral data.
# 
# To ease your start with the data you can read the feature descriptions below:
# 
# View "PhotoObj"
# objid = Object Identifier
# ra = J2000 Right Ascension (r-band)
# dec = J2000 Declination (r-band)
# Right ascension (abbreviated RA) is the angular distance measured eastward along the celestial equator from the Sun at the March equinox to the hour circle of the point above the earth in question. When paired with declination (abbreviated dec), these astronomical coordinates specify the direction of a point on the celestial sphere (traditionally called in English the skies or the sky) in the equatorial coordinate system.
# 
# Source: https://en.wikipedia.org/wiki/Right_ascension
# 
# u = better of DeV/Exp magnitude fit
# g = better of DeV/Exp magnitude fit
# r = better of DeV/Exp magnitude fit
# i = better of DeV/Exp magnitude fit
# z = better of DeV/Exp magnitude fit
# The Thuan-Gunn astronomic magnitude system. u, g, r, i, z represent the response of the 5 bands of the telescope.
# 
# Further education: https://www.astro.umd.edu/~ssm/ASTR620/mags.html
# 
# run = Run Number
# rereun = Rerun Number
# camcol = Camera column
# field = Field number
# Run, rerun, camcol and field are features which describe a field within an image taken by the SDSS. A field is basically a part of the entire image corresponding to 2048 by 1489 pixels. A field can be identified by: - run number, which identifies the specific scan, - the camera column, or "camcol," a number from 1 to 6, identifying the scanline within the run, and - the field number. The field number typically starts at 11 (after an initial rampup time), and can be as large as 800 for particularly long runs. - An additional number, rerun, specifies how the image was processed.
# 
# View "SpecObj"
# specobjid = Object Identifier
# class = object class (galaxy, star or quasar object)
# The class identifies an object to be either a galaxy, star or quasar. This will be the response variable which we will be trying to predict.
# 
# redshift = Final Redshift
# plate = plate number
# mjd = MJD of observation
# fiberid = fiber ID
# In physics, redshift happens when light or other electromagnetic radiation from an object is increased in wavelength, or shifted to the red end of the spectrum.
# 
# Each spectroscopic exposure employs a large, thin, circular metal plate that positions optical fibers via holes drilled at the locations of the images in the telescope focal plane. These fibers then feed into the spectrographs. Each plate has a unique serial number, which is called plate in views such as SpecObj in the CAS.
# 
# Modified Julian Date, used to indicate the date that a given piece of SDSS data (image or spectrum) was taken.
# 
# The SDSS spectrograph uses optical fibers to direct the light at the focal plane from individual objects to the slithead. Each object is assigned a corresponding fiberID.
# 
# Further information on SDSS images and their attributes:
# 
# http://www.sdss3.org/dr9/imaging/imaging_basics.php
# 
# http://www.sdss3.org/dr8/glossary.php
# 
# Acknowledgements
# The data released by the SDSS is under public domain. Its taken from the current data release RD14.
# 
# More information about the license:
# 
# http://www.sdss.org/science/image-gallery/
# 
# It was acquired by querying the CasJobs database which contains all data published by the SDSS.
# 
# The exact query can be found at:
# 
# http://skyserver.sdss.org/CasJobs/ (Free account is required!)
# 
# There are also other ways to get data from the SDSS catalogue. They can be found under:
# 
# http://www.sdss.org/dr14/
# 
# They really have a huge database which offers the possibility of creating all kinds of tables with respect to personal interests.
# 
# Please don't hesitate to contact me regarding any questions or improvement suggestions. :-)
# 
# Inspiration
# The dataset offers plenty of information about space to explore. Also the class column is the perfect target for classification practices!
# 
# Note: Since the data was already maintained very well it might not be best dataset to practice data cleaning / filtering...your decision though.

# Here I will build a random forest learning method to solve this problem. And at the end of this topic I will show the confusion matrix and classification report to show that how our problem is working.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')


# In[ ]:


df.drop('objid',axis = 1, inplace = True)
df.head()


# Now I will create a heatmap to see if there is any null value present or not.

# In[ ]:


plt.figure(figsize = (10,6))
sns.heatmap(df.isnull()==False)


# Woww! we can see from the above heatmap that no null values are there.
# Let's see whethere these values are int, float or string.

# In[ ]:


df.info()


# Now here is some exploretory data analysis for different attribute.

# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(y = 'ra', x = 'class', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.countplot(x = 'class', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'u', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'g', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'r', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'i', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'z', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'redshift', data = df)


# In[ ]:


plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'mjd', data = df)


# Now here I am doing train_test_split of the dataset and then I will create a Random Forest model to train our dataset and at the end i will show the prediction accuracy.

# In[ ]:


from sklearn.cross_validation import train_test_split
y = df['class']
x = df.drop('class', axis = 1)
x = x.drop('ra', axis = 1)
x = x.drop('dec', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 101)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(x_train, y_train)


# In[ ]:


pred = rfc.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test, pred))


# In[ ]:


print(confusion_matrix(y_test, pred))


# From the above confusion matrix prediction accuracy = (1495+212+1250)/(1495+9+10+24+212+1250) = 0.986
# 
# ***So from Above Random Forest method we can see it is performing very well in this dataset.***
# 
# *I am very new to machine learning. Feel free to leave comment if I did any mistake or how I can improve my machine learning model. Thank You.*
