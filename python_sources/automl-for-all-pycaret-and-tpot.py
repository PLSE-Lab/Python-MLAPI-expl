#!/usr/bin/env python
# coding: utf-8

# **Automated Machine Learning** or **AutoML** is the automation of Machine Learning algorithms which helps to find best algorithm/s that best fit the dataset in hand. With advancements in AutoML techniques, the life cycle of a data science process is reduced to a larger extent, making it easier for a company to derive efficient results by investing less time and workflow. It is considered that most Data Scientists spend their time in cleaning and organizing the data to structured formats ready for analysis and further process. Though these AutoML frameworks are not capable of these tasks, however, it helps a Data Science professional to increase his/her time to look for insights in the data provided or collected, by saving time needed for model building. 
# 
# 
# Advantages of Automated Machine Learning over traditional Model Development
# 1. Increases Productivity by automating repetitive tasks in Data science work flow, which in turn help data scientists and analysts to concentrate more on problem statement and extracting useful insights from data. Thus, one can save lot of time and invest the same in other tasks. 
# 1. 	Reduces the chances of errors when dealing with complex datasets with lot of advanced data-types and features.  Eg. ML models do not work well when dealing with categorical variables but the AutoML frameworks handles this like a piece of cake. 
# 1. 	Easier to learn for everyone and simple to apply for Developers into their projects in a shorter duration of time.
# 1. 	Useful in Hackathons, as one can find the find the optimal solution in a shorter span of time.
# 1. 	Deployment ready model can be developed with no risk.

# Steps invloved in PyCaret Model Building and Development
# 1. Installing and Importing PyCaret Library
# 2. Dataset Loading
# 3. Problem Statement understanding, Classfication or Regression
# 4. Loading data into PyCaret framework using setup
# 5. Find best models by training it with different algorithms.
# 6. Choose the best algorithm/s and hyper-tune the parameters for better performance.
# 7. Visualize the Results
# 8. Predictive Analytics using PyCaret
# 9. Deployment ready model development
# * **Note:** It hardly takes 6 lines of code to build a deployable predictive model

# In[ ]:


get_ipython().system(' pip install pycaret # Quite large depencies to install !')


# In[ ]:


import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


data = pd.read_csv('../input/titanic/train.csv')
data.head()


# In[ ]:


import pycaret
from pycaret.classification import *


# # Load and setup the dataset into PyCaret frame work

# In[ ]:


clf1 = setup(data = data, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['PassengerId','Name','Ticket','Cabin'],
             silent = True)


# # Compare and Select the Best Algorithm

# In[ ]:


compare_models()


# # Choose the Best performing Algorithm and Check for various Evaluations metrics

# In[ ]:


lgbm  = create_model('lightgbm')     


# # Hypertune the parameters using tune_model function

# In[ ]:


tuned_lgbm = tune_model('lightgbm')


# In[ ]:


plot_model(estimator = tuned_lgbm, plot = 'learning')


# In[ ]:


plot_model(estimator = tuned_lgbm, plot = 'feature')


# Confusion matrix at its ease!

# In[ ]:


plot_model(estimator = tuned_lgbm, plot = 'confusion_matrix')


# In[ ]:


# AUC Curve for Classifications models
plot_model(estimator = tuned_lgbm, plot = 'auc')


# In[ ]:


# Understand which feature had most role to play in the classification task
interpret_model(tuned_lgbm)


# Save the model you trained to a deployable pickle file (.pkl) using a simple line of code written below. 

# In[ ]:


save_model(tuned_lgbm, 'Titaniclgbm')
# code to load the model for future uses or when making predictions
# trained_model = load_model('Titaniclgbm')


# # Predictive Analysis using Test data

# In[ ]:


# Load the test data
test = pd.read_csv('../input/titanic/test.csv') 
predict_model(tuned_lgbm, data=test)


# In[ ]:


predictions = predict_model(tuned_lgbm, data=test)
predictions.head()


# In[ ]:


sub   = pd.read_csv('../input/titanic/gender_submission.csv')


# # Convert the predictions into structured dataframe, such as submission.csv

# In[ ]:


sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)


# In[ ]:


# Blend your model ton other algorithm.
xgb   = create_model('xgboost');    
logr  = create_model('lr');   
blend = blend_models(estimator_list=[tuned_lgbm,logr,xgb])


# # 2) TPOT AutoML (Tree Based Pipeline Optimization Tool)

# Import the Dataset as usual

# In[ ]:


data = pd.read_csv('../input/titanic/train.csv')
data.head()


# Find the sum of missing values in the dataset

# In[ ]:


data.isna().sum()


# Dropping some columns or features and Label Encoding features with categorical values. 

# In[ ]:


data =  data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data['Sex'] = le.fit_transform(data.Sex)


# In[ ]:


replacer = {'S':2,'C':1,'Q':0}
data['Embarked'] = data['Embarked'].map(replacer)
data.head()


# Train and test set splitting

# In[ ]:


train = data.drop('Survived',axis=1)
test = data['Survived']
train.shape, test.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train,test,test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# Import tpot classifier and fit train and test datasets
# Set up Hyper-parameters of the TPOT Classifer such max_time_mins, which states maximum time for training the model through iterations of generations.
tpot = TPOTClassifier(verbosity=2, max_time_mins=10)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))


# In[ ]:


# Check the specifications of the best fit algorith found out by TPOT
tpot.fitted_pipeline_


# In[ ]:


print(tpot.score(X_test, y_test))


# This is an extraoridnary feature of TPOT as it provides the code of the model as output. Here i exported the code of my model that best fits the problem statement. 

# In[ ]:


tpot.export('TPOTSOLN.py')
# Check the output dir of the Notebook to your top-right. Download it and as a surprise you will see python code of your model with perfectly tuned hyper-parameters. 


# # Kindly UPVOTE and Suggest your views and ideas ! Happy Learning !!
