#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Description 
# This notebook allows you to create a logistic regression model in order  to predict whether a particular internet user will click or not on an advertising. We will work with this [dataset](https://www.kaggle.com/fayomi/advertising)

# # Get the data
# * Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data to test and train data 
from sklearn.linear_model import LogisticRegression # Logistic Regression model
from sklearn.metrics import classification_report #evaluate the model
import matplotlib.pyplot as plt #visualisation 


# * Read the cvs file and set it to a dataframe called ad_data

# In[ ]:


#get the data path
ad_path="../input/advertising/advertising.csv"

#get the data
ad_data=pd.read_csv(ad_path)

#show the 5 rows of ad_data
ad_data.head()


# * show statistics information about numerical columns

# In[ ]:


ad_data.describe()


# # Logistic Regression model
# * Chose the features 

# In[ ]:


#The outcome 
y=ad_data["Clicked on Ad"]
# The features 
X= ad_data[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]


# * Split the data

# In[ ]:


#33% of data is used for test the rest of data is used for training our model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# * Train and fit a logistic regression model on the training set

# In[ ]:


#create an instance of Logistic regression model 
lgmodel=LogisticRegression()

#fit the model on the training set
lgmodel.fit(X_train,y_train)


# # Prediction & Evaluations
# * Prediction using the testing set

# In[ ]:


predictions =lgmodel.predict(X_test)


# * Evaluation

# In[ ]:


#Print a text report showing the main classification metrics
print(classification_report(y_test,predictions))


# Here is the [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) of the classification report 

# # Bonus
# Fit the model on the entire data and save the predictions on csv file 

# In[ ]:


final_predictions = lgmodel.predict(X)


# In[ ]:


predic = pd.DataFrame({'Predict_Click on Ad': final_predictions})
output=pd.concat([X,predic], axis=1)
output.head()


# In[ ]:


output.to_csv('Ad_predictions.csv', index=False)
print("Your output was successfully saved!")

