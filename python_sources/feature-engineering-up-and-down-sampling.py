#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

import numpy as np


# calculate accuracy measures and confusion matrix
from sklearn import metrics


from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the dataset
pima_df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


pima_df.head()


# In[ ]:


# Let us check whether any of the columns has any value other than numeric i.e. data is not corrupted such as a "?" instead of 
# a number.

# we use np.isreal a numpy function which checks each column for each row and returns a bool array, 
# where True if input element is real.
# applymap is pandas dataframe function that applies the np.isreal function columnwise
# Following line selects those rows which have some non-numeric value in any of the columns hence the  ~ symbol

pima_df[~pima_df.applymap(np.isreal).all(1)]


# In[ ]:


# replace the missing values in pima_df with median value :Note, we do not need to specify the column names
# every column's missing value is replaced with that column's median respectively
pima_df = pima_df.fillna(pima_df.median())
pima_df


# In[ ]:


#Lets analysze the distribution of the various attributes
pima_df.describe().transpose()


# In[ ]:


# Let us look at the target column which is 'Outcome' to understand how the data is distributed amongst the various values
pima_df.groupby(["Outcome"]).count()

# Most are not diabetic. The ratio is almost 1:2 in favor or class 0.  The model's ability to predict class 0 will 
# be better than predicting class 1. 


# In[ ]:


# Pairplot using sns

sns.pairplot(pima_df , hue='Outcome' , diag_kind = 'kde')


# In[ ]:


#data for all the attributes are skewed, especially for the variable "Insulin"

#The mean for test is 80(rounded) while the median is 30.5 which clearly indicates an extreme long tail on the right


# In[ ]:


# Attributes which look normally distributed (glucose, blood pressure, skin thickness, and BMI).
# Some of the attributes look like they may have an exponential distribution (pregnancy, insulin, DiabetesPedigreeFunction, age).
# Age should probably have a normal distribution, the constraints on the data collection may have skewed the distribution.

# There is no obvious relationship between age and onset of diabetes.
# There is no obvious relationship between DiabetesPedigreeFunction function and onset of diabetes.


# In[ ]:


array = pima_df.values
X = array[:,0:7] # select all rows and first 8 columns which are the attributes
Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# # SMOTE to upsample smaller class

# In[ ]:


print("Before UpSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before UpSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(sampling_strategy = 1 ,k_neighbors = 5, random_state=1)   #Synthetic Minority Over Sampling Technique
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


print("After UpSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UpSampling, counts of label '0': {} \n".format(sum(y_train_res==0)))



print('After UpSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_res.shape))


# In[ ]:


# Fit the model on original data i.e. before upsampling

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)


# In[ ]:


test_pred = model.predict(X_test)

print(metrics.classification_report(y_test, test_pred))
print(metrics.confusion_matrix(y_test, test_pred))


# # UpSample smaller class

# In[ ]:


# fit model on upsampled data 

model.fit(X_train_res, y_train_res)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))


# # Down Sampling the larger class

# In[ ]:


non_diab_indices = pima_df[pima_df['Outcome'] == 0].index   # Get the record numbers of non-diab cases
no_diab = len(pima_df[pima_df['Outcome'] == 0])             # how many non-diab cases
print(no_diab)

diab_indices = pima_df[pima_df['Outcome'] == 1].index       # record number of the diabeteics cases
diab = len(pima_df[pima_df['Outcome'] == 1])                # how many diabetic cases
print(diab)


# In[ ]:


random_indices = np.random.choice( non_diab_indices, no_diab - 200 , replace=False)    #Randomly pick up 200 non-diab indices


# In[ ]:


down_sample_indices = np.concatenate([diab_indices,random_indices])  # combine the 200 non-diab indices with diab indices


# In[ ]:


pima_df_down_sample = pima_df.loc[down_sample_indices]  # Extract all those records for diab and non-diab to create new set
pima_df_down_sample.shape
pima_df_down_sample.groupby(["Outcome"]).count()  # look at the class distribution after downsample


# In[ ]:


array = pima_df_down_sample.values
X = array[:,0:7] # select all rows and first 8 columns which are the attributes
Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# In[ ]:


print('After DownSampling, the shape of X_train: {}'.format(X_train.shape))
print('After DownSampling, the shape of X_test: {} \n'.format(X_test.shape))


# In[ ]:


# Fit the model on 30%
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))


# ## IMBLearn Random Under Sampling

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


rus = RandomUnderSampler(return_indices=True)


# In[ ]:


X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)


# In[ ]:


y_rus


# In[ ]:


y_rus.shape


# ## IMBLearn Random Over Sampling
# 

# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X_train, y_train)


# In[ ]:


y_ros


# In[ ]:


y_ros.shape


# In[ ]:


X_ros.shape


# ##  Deleting nearest majority neighbors  TomekLinks
# 

# In[ ]:


from imblearn.under_sampling import TomekLinkstl = TomekLinks(return_indices=True, ratio='majority')


# In[ ]:


tl = TomekLinks(return_indices=True, ratio='majority')


# In[ ]:


X_tl, y_tl, id_tl = tl.fit_sample(X_train, y_train)   # id_tl is removed instances of majority class


# In[ ]:


y_tl.shape


# In[ ]:


X_tl.shape


# ## Upsampling followed by downsampling
# 

# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


smt = SMOTETomek(ratio='auto')


# In[ ]:


X_smt, y_smt = smt.fit_sample(X_train, y_train)


# In[ ]:


X_smt.shape


# ## Cluster based undersampling
# 

# In[ ]:


from imblearn.under_sampling import ClusterCentroids


# In[ ]:


cc = ClusterCentroids()  
X_cc, y_cc = cc.fit_sample(X_train, y_train)


# In[ ]:


X_cc.shape


# In[ ]:


y_cc

