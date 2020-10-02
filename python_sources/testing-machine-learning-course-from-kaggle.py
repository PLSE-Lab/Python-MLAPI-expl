#!/usr/bin/env python
# coding: utf-8

# 1. [Introduction](#Introduction) 
# 2. [Import Packages](#Packages)
# 3. [Analyzing the Data](#Data)
# 4. [Building the Model](#Model)
# 5. [Model Validation](#Error)
# 

# <div id="Introduction">**1. Introduction** 
# 
# 
# Just learned the python package, scikit-learn. I want to test what I learned to predict graduate admissions from the Graduate Admissions dataset. The micro-course I used to learn machine learning was from Kaggle.
# 
# [Here is the link for the Kaggle's micro-course for Machine Learning](http://https://www.kaggle.com/learn/machine-learning)
# 

# <div id="Packages">**2. Import Packages**
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# <div id="Data">**3. Analyzing the Data**
# 
# First we upload the dataset. Afterwards, we test to see if the dataset runs properly and then we analyze what is in the dataset. 

# In[ ]:


# Uploading the dataset
graduate_admisssion_data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


# Test to see if the data can open
graduate_admisssion_data.head()


# In[ ]:


# Quick analzation of the dataset
graduate_admisssion_data.describe()


# From the data, we can see that there are 500 entries. The columns that matter are GRE(Graduate Record Examination) scores, TOEFL(Test of English as a Foreign Language) scores, the University rating, SOP(Statement of Purpose), LOR(Letter of Recommendation), CGPA(Undergraduate Grade Point Average), if the individual has done research or not, and chance of admission. 
# 
# For the GRE score the mean was 316 points, with a standard deviation of +/-11 points. The lowest score in the data was 290 points, while the highest was 340. Is important to point out that the lowest combined score possible on the GRE is 260 points and the maximum possible points is 340 points. 
# 
# For the TOEFL score the mean was 107 points, with a standard deviation of +/-6 points. The lowest score in the data was 92 points and the highest was 120. Unlike the GRE, the lowest possible score on the TOEFL exam is 0 points. While the highest possible points was 120. 
# 
# For Unviersity rating, the score from 1 to 5.
# 
# The same scoring as University rating is applied to SOP and LOR.
# 
# Similar to SOP and the University rating, LOR scoring is from 1 to 5.
# 
# The Undergraduate GPA on the other hand is rated from a 10 point scale, 10 being the highest. The mean GPA was scored as 8.58, with a standard deviation of .60 points. The lowest GPA in the dataset was 6.8 and the highest was 9.92.  
# 
# For research experience, 0 means an individual has **not** done research and 1 means that an individual has done research. Most candidates have done past research, as seen from the dataset.
# 
# Lastly, an individual chance of admission is base off of a percentage. The mean chance of an individual getting accepted was 72% with a +/-14% standard deviation. One individual lowest possibility was 34% and another was 97%.  
# 
# 

# <div id=Model></dev>**4. Building the Model**
# 
# As stated earlier, we only need 8 out of the 9 columns. We will use the seven columns, GRE score, TOEFL score, University rating, SOP, LOR, GPA, and research experience. These columns will be our predictors, while the chance of admission will be the prediction target.   
# 
# 

# In[ ]:


# Check if the columns name are correctly named
graduate_admisssion_data.columns


# In[ ]:


# Clean the titles of the columns, because there are spaces
graduate_admisssion_data.columns = graduate_admisssion_data.columns.str.replace(' ', '_')
#Check to see if the spaces have been replaced
graduate_admisssion_data.columns


# In[ ]:


y = graduate_admisssion_data.Chance_of_Admit_
admission_features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR_', 'CGPA', 'Research']
X = graduate_admisssion_data[admission_features]


# In[ ]:


admission_model = DecisionTreeRegressor(random_state=1)
admission_model.fit(X, y)


# In[ ]:


print("Making predictions for the following 5 students:")
print(X.head())
print("The predictions are")
print(admission_model.predict(X.head()))


# <div id='Error'>**5. Model Validation**
# 
# We want to see how accuract the model predicts are.
# 

# In[ ]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return(mae)


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
admission_model = DecisionTreeRegressor()
admission_model.fit(train_X, train_y)
val_predictions = admission_model.predict(val_X)
print (mean_absolute_error(val_y, val_predictions))


# 
