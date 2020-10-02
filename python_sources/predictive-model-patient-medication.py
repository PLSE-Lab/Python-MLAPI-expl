#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# As part of defining the right medicine and treatment for patients with same illness, we can create a prediction model by leveraging patient data and their response to a certain medication. Our goal in this project is to create a model to find the right medication for future patients with same illness. The problem that is in question which we are trying to solve is "Can we create a machine learning algorithm where we can predict proper medication to a group of patient with same illness, based on their features such as age and gender?" 

# # Contents
# 
# 1. About the Data Set
# 2. Data Collection and Understanding
# 3. Data Exploration
# 4. Model Selecting and Set Up
# 5. Model Development
# 6. Prediction
# 7. Evaluation
# 8. Conclusion

# ## About the Data Set
# 
# The features that are provided within the data set are outlined as below
# 
# - Age : Age of the Patient
# - Sex : Gender of the Patient
# - BP  : Blood Pressure of the Patient
# - Cholesterol: Cholesterol of the Patient
# - Drug: Drug each patient responded to
# - Na_to_K: Sodium to Potasium Levels
# 
# 
# * Please note all patients in the dataset have the same illness.

# ## Data Collection and Understanding

# In[ ]:


# importing neccessary libraries
import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv('../input/drug200.csv')
df.head(5)


# In[ ]:


# summary of the dataframe
df.info()


# We can see that the data set size is 1200 with 200 rows and 6 columns. The variables are correct data type; Age is integer, Sex, BP, Cholesterol and Drug is objects, Na_to_K is float.

# We can also see that Sex, BP and Cholesterol are categorical variables. 

# In[ ]:


# looking to see if there are any missing values
missing_data=df.isnull()
missing_data.head(5)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')


# We can see that there are no missing values within our dataset. We can start analyzing our data set. 

# ## Data Exploration

# In[ ]:


df.describe()


# Based on the data set, our average age of patients is 44. The youngest patient is 15 and oldest patient is 74 years old. Please keep in mind that all of the patients have the same illness. 

# In[ ]:


df.corr()


# There is a negative medium correlation between the age and, Sodium to Potassium ratio. The higher the age is the lower the sodium to potassium ratio is.

# ## Model Selecting and Set Up
# 
# Based on the feature data set, even though we do have categorical variables, we can use Decision Tree to create a prediction model. In order to do that, we need to change the categorical variables such as Sex, Blood Pressure and Cholesterol to numerical variables.
# 
# We can define our Feature Matrix is as X and y as the response vector which is the target.

# In[ ]:


# defining our feature matrix that will predict the target y value(drug)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[ ]:


# turn the categorical variables to numeric variables
from sklearn import preprocessing


# In[ ]:


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[ ]:


# defining the y target value
y=df['Drug']
y[0:5]


# We have selected Decision Tree as predictive modeling, defined our feature matrix and target. We can start setting up the decision tree.

# In[ ]:


# importing neccessary libraries
from sklearn.model_selection import train_test_split


# In[ ]:


X_trainset, X_testset, y_trainset, y_testset=train_test_split(X, y, test_size=0.3, random_state=3)


# In[ ]:


X_trainset[0:5]


# ## Model Development

# In[ ]:


# creating the Decision Tree Clasifier instance
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[ ]:


# we will fit the X, y trainset. (training the dataset with X, y trainset values)
drugTree.fit(X_trainset,y_trainset)


# ## Prediction

# In[ ]:


# our model is ready and we can start defining predictions
predTree = drugTree.predict(X_testset)


# In[ ]:


print (predTree [0:5])
print (y_testset [0:5])


# ## Evaluation

# In[ ]:


# our model is ready and we can check the accuracy of the model by importing metrics from sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# The classification score we get is based on: a set of labels predicted for a sample must exactlty match the corresponding set of labels. 

# ## Conclusion
# 
# Based on our analysis, by using the Age, Sex, Blood Pressure, Cholesterol and Sodium to Potasium ratio as feature matrix, we created a machine learning model to predict proper medication for a patient. 

# In[ ]:




