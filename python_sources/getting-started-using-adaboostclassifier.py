#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This is a very simple kernel to bring a person onboard for Pet Finder - Competition
# 
# AdaBoostClassifier is used with Python for the predictions. It helps you to get started with this competition. Further Analysis and enhancements can be built over this kernel. Feel free to fork and use it as per your needs.
# 
# 
# **Steps:-**
# 
# **Step 1.** Load the required Python Modules

# In[ ]:


# Loading the required python modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


# **Step 2. **Read and Load the Test and Train Data

# In[ ]:


# Reading Training Data
trainDataCsvFilepath = '../input/train/train.csv'
trainDataFrame = pd.read_csv(trainDataCsvFilepath)

# Reading Test Data
testDataCsvFilepath = '../input/test/test.csv'
testDataFrame = pd.read_csv(testDataCsvFilepath)


# **Step 3.** List out the Most Significant Features of the Data. Then, Create the Training DataFrames and Test DataFrames using the features list.

# In[ ]:


# Features List which are being targeted
featuresList = ['Age','Health','Vaccinated','Dewormed','Sterilized','PhotoAmt','Gender','Breed1','Breed2','Color1','Color2','Fee','MaturitySize']
x = trainDataFrame[featuresList]
y = trainDataFrame.AdoptionSpeed

# DataFrame for testing the Prediction Model
test_x = testDataFrame[featuresList]


# **Step 4.** Use AdaBoostClassifier to generate predictions on Test Dataset and save predictions to submission.csv file

# In[ ]:


#Prediction using AdaBoostClassifier
clf = AdaBoostClassifier()

# Load Training Dataset into the Classifier
clf.fit(x, y)
pred = pd.DataFrame()
pred['PetID'] = testDataFrame['PetID']

# Generate Prediction on Test Dataset using the Trained Model
pred['AdoptionSpeed'] = clf.predict(test_x)

# Saving Predicitions to submission.csv file
pred.set_index('PetID').to_csv("submission.csv", index=True)


# 
