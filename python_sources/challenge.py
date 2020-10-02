#!/usr/bin/env python
# coding: utf-8

# # Data challenge @ EDHEC: personal injury accidents
# ## Context
# For each personal injury accident (i.e. an accident occurring on a road open to public traffic, involving at least one vehicle and resulting in at least one victim requiring treatment), information describing the accident is entered by the police unit that intervened at the accident site. These entries are collected in the national road traffic injury file.
# 
# The present database lists all traffic accidents that occurred during a given year in metropolitan France and the overseas departments (Guadeloupe, French Guiana, Martinique, Reunion Island and Mayotte) with a simplified description. This includes information on the location of the accident, such as information on the characteristics of the accident and its location, the vehicles involved and their victims.
# 
# ## Dataset
# The dataset contains thousands of individuals and 32 variables (which are either quantitative or **categorical**) with many missing values. The description of these variables is available in the file *description.txt*.
# 
# The target variable is the severity of each personal injury accident.
# It is a label taking 4 values:
# 
#  1. Unscathed
#  2. Slight injury
#  3. Injured in hospital
#  4. Killed
# 
# ## Goal
# The goal of this challenge is to predict the severity of a personal injury accident based on 31 descriptive variables.
# 
# 
# 
# 
# # Evaluation
# ## Evaluation metric
# The evaluation metric si the categorization accuracy (i.e. the usual classification score).
# 
# ## Submission Format
# An example of submission file is provided in the file *challenge.ipynb* (see Kernels section).
# 
# 
# 
# 
# # Data
# ## File descriptions
# - Xtrain.csv: matrix of input data for training (each individual is a row);
# - ytrain.csv: vector of input targets for training (it has as many rows as Xtrain.csv);
# - Xtest.csv: matrix of input data for prediction (each individual is a row);
# - description.txt: description of variables.
# 
# ## Target
# The target is the severity of each personal injury accident.
# It can take four different values:
# 
#  1. Unscathed
#  2. Slight injury
#  3. Injured in hospital
#  4. Killed
# 
# ## Data fields
# See the file *description.txt* for a detailed description.
# 
# **month**: 
# Month of the accident.
# 
# **day**: 
# Day of the accident.
# 
# **time**: 
# Hours and minutes of the accident.
# 
# **light**: 
# Lighting conditions in which the accident occurred.
# 
# **location**: 
# Location.
# 
# **intersection**: 
# Intersection.
# 
# **atmosphere**: 
# Weather conditions.
# 
# **collision**: 
# Type of collision:
# 
# **municipality**: 
# The municipality number is a code given by INSEE. The code has 3 digits on the right.
# 
# **department**: 
# INSEE code (Institut National de la Statistique et des Etudes Economiques) of the monitoring department.
# 
# **road**: 
# Road category.
# 
# **traffic**: 
# Traffic regime.
# 
# **nblanes**: 
# Total number of lanes of traffic.
# 
# **reslane**: 
# Indicates the existence of a reserved lane, regardless of whether or not the accident occurs on that lane.
# 
# **laneprofile**: 
# Long profile describes the gradient of the road at the accident site.
# 
# **plan**: 
# Drawing in plan:
# 
# **surface**: 
# Surface condition.
# 
# **user**: 
# User category.
# 
# **sex**: 
# Gender of the user.
# 
# **route**: 
# Reason for travel at the time of the accident.
# 
# **pedlocation**: 
# Pedestrian location.
# 
# **pedaction**: 
# Pedestrian action.
# 
# **pedcondition**: 
# This variable is used to specify whether the injured pedestrian was alone or not.
# 
# **birth**: 
# User's year of birth.
# 
# **dirtraffic**: 
# Direction of traffic.
# 
# **catvehicle**: 
# Vehicle category.
# 
# **nboccupants**: 
# Number of occupants in public transit.
# 
# **obstacle**: 
# Fixed obstacle hit.
# 
# **movobstacle**: 
# Mobile obstacle hit.
# 
# **impact**: 
# Initial shock point.
# 
# **maneuver**: 
# Main manoeuvre before the accident.
# 
# 
# 
# 
# # Rules
# The examination linked to this challenge will be based on four components:
#  1. the Kaggle leaderboard (the best, the better mark);
#  2. a concise Python notebook enabling to rebuild your best model;
#  3. a slideshow (about 15 slides), presenting your approach (method, evaluation procedure, achievements, pitfalls);
#  4. a 15-minute defense supported by your slideshow (**minor Data sciences only**).
# 
# ## Kaggle rules
# ### Leaderboard
# The purpose of the Kaggle leaderboard is to:
# 
#  - assess the ranking of each team (this is the public leaderboard);
#  - produce the final ranking used for the exam (this is the private leaderboard).
# 
# In order to use the Kaggle leaderboard, you have to upload a prediction file.
# This latter should have an 'id' column and a 'prediction' one, similarly to the example provided below (see Overview > Evaluation).
# 
# This is a **challenge**: your main goal is to stand at the top of the ranking.
# Thus, you are expected to submit **several** times.
# 
# ### One account per participant
# Each participant has to sign up to Kaggle with his real name.
# You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.
# 
# ### No private sharing outside teams
# Privately sharing code or data outside of teams is not permitted. It's okay to share code if made available to all participants on the forums.
# 
# ### Team work
# You have to work and join the challenge in teams of 3 participants **enrolled in the same minor** (Business analytics or Data sciences).
# 
# ### Team mergers
# Team mergers are not allowed in this competition.
# 
# ### Submission limits
# You may submit a maximum of 5 entries per day.
# 
# You may select up to 2 final submissions for judging.
# 
# ### Competition timeline
# 
# Start Date: **April 17**
# 
# End Date: **May 21**
# 
# ## Notebook
# In order to assess your ease in data science with Python, you have to upload a concise notebook (both *ipynb* and *html* files) enabling to rebuild your best model to this remote repository: https://www.dropbox.com/request/NJ55wvnUqhP3eam8Nik4
# 
# Due: **May 21**
# 
# ## Slideshow
# You have to prepare a slideshow (about 15 slides). It is aimed at presenting your approach, including:
# 
#  - a quick explanation of the methods selected and rejected;
#  - your evaluation method (parameter selection);
#  - your achievements;
#  - the pitfalls encountered.
# 
# You have to upload your slides to this remote repository: https://www.dropbox.com/request/NJ55wvnUqhP3eam8Nik4
# 
# Due: **May 27**
# 
# ## Defense (minor Data sciences only)
# The final step of the exam will be a 15-minute defense supported by a slideshow (see previous section). You will have to answer general and pratical questions at the end of your talk.
# 
# Date: **May 28**
# 

# In[1]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Loading the data
# Here we have:
# - `X_train`: matrix of input data for training (each individual is a row);
# - `y_train`: vector of input labels for training (it has as many rows as `X_train`);
# - `X_test`: matrix of input data for prediction (each individual is a row).

# In[2]:


data_train = read_csv('../input/Xtrain.csv', header=0)  # Training input data
label_train = read_csv('../input/ytrain.csv', header=None)  # Training labels
data_test = read_csv('../input/Xtest.csv', header=0)  # Test input data


# Input variables may have missing values:

# In[3]:


concat([data_train.dtypes, data_train.isna().any(axis=0)], keys=['type', 'missing value?'], axis=1, sort=False)


# Here, we fill missing values in a very naive way.

# In[4]:


data_train = data_train.fillna(value=0)  # Fill missing valxues with 0
data_test = data_test.fillna(value=0)  # Fill missing values with 0


# The complete data set (containing the target variable `severity`) looks like this:

# In[5]:


data = data_train.copy()
data['severity'] = label_train[0]
data.head()


# The variable `severity` is categorical with 4 possible values:
#  1. Unscathed
#  2. Slight injury
#  3. Injured in hospital
#  4. Killed

# In[6]:


plt.figure(figsize=(10, 6))
sns.distplot(data['severity']);


# Of course, the distribution of this random variable differs when taken given the road category, the user category or the pedestrian location.

# In[7]:


plt.figure(figsize=(15, 7))
sns.violinplot(y='severity', x='road', data=data)


# In[8]:


plt.figure(figsize=(15, 7))
sns.violinplot(y='severity', x='user', data=data)


# In[9]:


plt.figure(figsize=(15, 7))
sns.violinplot(y='severity', x='pedlocation', data=data)


# Now, for the purpose of the challenge, we extract data as Numpy arrays.

# In[10]:


X_train = data_train.values  # Training input data as Numpy array
y_train = label_train.values.ravel()  # Training labels as Numpy array
X_test = data_test.values  # Test input data as Numpy array


# In[11]:


print('Shape of the training dataset:', X_train.shape)
print('Shape of the test dataset:', X_test.shape)


# Here, we provide a naive classifier.

# In[12]:


clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
print('Tree score:', clf.score(X_train, y_train))


# Once the classifier is fitted, you have to predict categories for the test dataset and save it on your hardrive.

# In[13]:


y_pred = clf.predict(X_test)  # Label predictions for the test set
DataFrame(y_pred).to_csv('ypred.csv', index_label='id', header=['prediction'])  # Save prediction


# The last step is to upload the file `ypred.csv` to the Kaggle leaderboard.
