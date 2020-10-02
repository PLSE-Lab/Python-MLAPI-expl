#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Downloading-and-Submitting-Data" data-toc-modified-id="Downloading-and-Submitting-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Downloading and Submitting Data</a></span><ul class="toc-item"><li><span><a href="#Setup" data-toc-modified-id="Setup-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Setup</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Data</a></span></li></ul></li><li><span><a href="#Understanding-the-data" data-toc-modified-id="Understanding-the-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Understanding the data</a></span></li><li><span><a href="#Data-preparation-and-machine-learning" data-toc-modified-id="Data-preparation-and-machine-learning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data preparation and machine learning</a></span></li><li><span><a href="#Submission" data-toc-modified-id="Submission-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Submission</a></span></li></ul></div>

# Here's a simple getting started notebook that shows you how to load the data, and how to create a Kaggle submission file. Remember that you should structure your notebook after the 8 step guide, as detailed in the [Assignment 1 instructions](https://hvl.instructure.com/courses/9086/assignments/17277). 

# # Downloading and Submitting Data

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


# ## Data

# Go to Kaggle competition website and download the data. Make a new folder in your DAT158ML repository called 'data'. Store the Kaggle competition data in this folder

# Then you should uncomment the code and run the following two cells. **Warning:** This doesn't work in this Kaggle hosted notebook! See below

# In[ ]:


#lists the files in the folder
#import os
#print(os.listdir("data"))


# In[ ]:


#Reads in the csv-files and creates a dataframe using pandas

#train = pd.read_csv('data/housing_data.csv')
#test = pd.read_csv('data/housing_test_data.csv')
#sampleSubmission = pd.read_csv('data/sample_submission.csv')


# **Kaggle-specific way of accessing the data**

# On Kaggle the data is stored in the folder `../input/dat158-2019/`:

# In[ ]:


train = pd.read_csv('../input/dat158-2019/housing_data.csv')
test = pd.read_csv('../input/dat158-2019/housing_test_data.csv')
sampleSubmission = pd.read_csv('../input/dat158-2019/sample_submission.csv')


# # Understanding the data

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# # Data preparation and machine learning

# This part you should code and figure out yourself. Play around with different ways to prepare the data, different machine learning models and settings of hyperparameters
# 
# Remember to create your own validation set to evaluate your models. Your test set will not contain labels and are therefore not suited for evaluating and tuning your different models. 
# 

# # Submission

# After you have trained your model and have found predictions on your test data, you must create a csv-file that contains 'Id' and your predictions in two coloums
# 
# We have assumed that you have called your predicitons 'median_house_value' after you have trained your model
# 
# This is just for demonstrational purposes, that is why all our predictions is zero. Yours will be filled with numbers
# 

# In[ ]:


median_house_value = [0 for i in test['Id']]


# In[ ]:


len(median_house_value)


# In[ ]:


median_house_value[:10]


# In[ ]:


submission = pd.DataFrame({'Id': test['Id'], 'median_house_value': median_house_value})


# In[ ]:


submission.head()


# In[ ]:


# Stores a csv file to submit to the kaggle competition
#submission.to_csv('submission.csv', index=False)

