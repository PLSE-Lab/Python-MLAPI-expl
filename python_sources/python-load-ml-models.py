#!/usr/bin/env python
# coding: utf-8

# **WHAT**
# On various instances, while working on developing a Machine Learning Model, we'll need to save our prediction models to file, and then restore them in order to reuse our previous work to.
# 
# 
# **WHY**
# We need to save and restore/reload later our ML Model , so as to -
# 
# a) test our model on/with new data, 
# 
# b) compare multiple models, 
# 
# c) or anything else. 
# 
# **object serialization**
# This process / procedure of saving a ML Model is also known as object serialization - representing an object with a stream of bytes, in order to store it on disk, send it over a network or save to a database.
# 
# **deserialization**
# While the restoring/reloading of ML Model procedure is known as deserialization. 
# 
# In this Kernel, we will explore 3 ways to Save and Reload ML Models in Python and scikit-learn, we will also discuss about the pros and cons of each method. 

# We will be covering following 3 approaches of Saving and Reloading a ML Model -
# 
# 1) Pickle Approach
# 
# 2) Joblib Approach
# 
# 3) Manual Save and Restore to JSON approach

# Now , lets develop a ML Model which we shall use to Save and Reload in this Kernel

# **ML Model Creation**
# 
# For the purpose of Demo , we will create a basic Logistic Regression Model on IRIS Dataset.
# 
# Dataset used : IRIS 
# 
# Model        : Logistic Regression using Scikit Learn

# **Step - 1 ** : Import Packages

# In[ ]:


# Import Required packages 
#-------------------------

# Import the Logistic Regression Module from Scikit Learn
from sklearn.linear_model import LogisticRegression  

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris  

# Load the Module to split the Dataset into Train & Test 
from sklearn.model_selection import train_test_split


# **Step - 2 **: Load the IRIS Data

# In[ ]:


# Load the data
Iris_data = load_iris()  


# **Step - 3 **: Split the IRIS Data into Training & Testing Data

# In[ ]:


# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data, 
                                                Iris_data.target, 
                                                test_size=0.3, 
                                                random_state=4)  


# Now , lets build the Logistic Regression Model on the IRIS Data
# 
# Note : The Model creation in this Kernel is for demonstartion only and does not cover the details of Model Creation.

# In[ ]:


# Define the Model
LR_Model = LogisticRegression(C=0.1,  
                               max_iter=20, 
                               fit_intercept=True, 
                               n_jobs=3, 
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)  


# Now , that Model has been Created and Trained , we might want to save the trained Model for future use.
# 

# **Approach 1 : Pickle approach**
# 
# Following lines of code, the LR_Model which we created in the previous step is saved to file, and then loaded as a new object called Pickled_RL_Model. 
# 
# The loaded model is then used to calculate the accuracy score and predict outcomes on new unseen (test) data.

# In[ ]:


# Import pickle Package

import pickle


# In[ ]:


# Save the Modle to file in the current working directory

Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LR_Model, file)


# In[ ]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model


# In[ ]:


# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_LR_Model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(Xtest)  

Ypredict


# **Let's Reflect back on Pickle approach :**
# 
# PROs of Pickle :
# 
# 1) save and restore our learning models is quick - we can do it in two lines of code. 
# 
# 2) It is useful if you have optimized the model's parameters on the training data, so you don't need to repeat this step again. 
# 
# 
# CONs of Pickle :
# 
# 1) it doesn't save the test results or any data. 
# 

# **Approach 2 - Joblib** :
# 
# The Joblib Module is available from Scikit Learn package and is intended to be a replacement for Pickle, for objects containing large data. 
# 
# This approach will save our ML Model in the pickle format only but we dont need to load additional libraries as the 'Pickling' facility is available within Scikit Learn package itself which we will use invariably for developing our ML models.
# 
# In following Python scripts , we will show how to Saev and reload ML Models using Joblib

# Import the required Library for using Joblib

# In[ ]:


# Import Joblib Module from Scikit Learn

from sklearn.externals import joblib


# Save the Model using Joblib

# In[ ]:


# Save RL_Model to file in the current working directory

joblib_file = "joblib_RL_Model.pkl"  
joblib.dump(LR_Model, joblib_file)


# Reload the saved Model using Joblib

# In[ ]:


# Load from file

joblib_LR_model = joblib.load(joblib_file)


joblib_LR_model


# Reload the Saved Model using Joblib 

# In[ ]:


# Use the Reloaded Joblib Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = joblib_LR_model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = joblib_LR_model.predict(Xtest)  

Ypredict


# **Let's Reflect back on Joblib approach :**
# 
# PROs of Joblib :
# 
# 1) the Joblib library offers a bit simpler workflow compared to Pickle. 
# 
# 2) While Pickle requires a file object to be passed as an argument, Joblib works with both file objects and string filenames. 
# 
# 3) In case our model contains large arrays of data, each array will be stored in a separate file, but the save and restore procedure will remain the same. 
# 
# 4) Joblib also allows different compression methods, such as 'zlib', 'gzip', 'bz2', and different levels of compression.
# 

# **Approach 3 - Manual Save and Restore to JSON ** :
# 
# whenever we want to have full control over the save and restore process, the best way is to build our own functions manually.
# 
# The Script following shows an example of manually saving and restoring objects using JSON. This approach allows us to select the data which needs to be saved, such as the model parameters, coefficients, training data, and anything else we need.
# 
# For simplicity, we'll save only three model parameters and the training data. Some additional data we could store with this approach is, for example, a cross-validation score on the training set, test data, accuracy score on the test data, etc.

# Import the required libraries

# In[ ]:


# Import required packages

import json  
import numpy as np


# Since we want to save all of this data in a single object, one possible way to do it is to create a new class which inherits from the model class, which in our example is LogisticRegression. The new class, called MyLogReg, then implements the methods save_json and load_json for saving and restoring to/from a JSON file, respectively.

# In[ ]:


class MyLogReg(LogisticRegression):

    # Override the class constructor
    def __init__(self, C=1.0, solver='liblinear', max_iter=100, X_train=None, Y_train=None):
        LogisticRegression.__init__(self, C=C, solver=solver, max_iter=max_iter)
        self.X_train = X_train
        self.Y_train = Y_train

    # A method for saving object data to JSON file
    def save_json(self, filepath):
        dict_ = {}
        dict_['C'] = self.C
        dict_['max_iter'] = self.max_iter
        dict_['solver'] = self.solver
        dict_['X_train'] = self.X_train.tolist() if self.X_train is not None else 'None'
        dict_['Y_train'] = self.Y_train.tolist() if self.Y_train is not None else 'None'

        # Creat json and save to file
        json_txt = json.dumps(dict_, indent=4)
        with open(filepath, 'w') as file:
            file.write(json_txt)

    # A method for loading data from JSON file
    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            dict_ = json.load(file)

        self.C = dict_['C']
        self.max_iter = dict_['max_iter']
        self.solver = dict_['solver']
        self.X_train = np.asarray(dict_['X_train']) if dict_['X_train'] != 'None' else None
        self.Y_train = np.asarray(dict_['Y_train']) if dict_['Y_train'] != 'None' else None


# Next we create an object mylogreg, pass the training data to it, and save it to file. 
# 
# Then we create a new object json_mylogreg and call the load_json method to load the data from file.

# In[ ]:


filepath = "mylogreg.json"

# Create a model and train it
mylogreg = MyLogReg(X_train=Xtrain, Y_train=Ytrain)  
mylogreg.save_json(filepath)

# Create a new object and load its data from JSON file
json_mylogreg = MyLogReg()  
json_mylogreg.load_json(filepath)  
json_mylogreg  


# **Let's reflect back on the JSON approach**
# 
# PROs :
# 
# Since the data serialization using JSON actually saves the object into a string format, rather than byte stream, the 'mylogreg.json' file could be opened and modified with a text editor.
# 
# 
# CONs :
# 
# Although this approach would be convenient for the developer, it is less secure since an intruder can view and amend the content of the JSON file. 
# 
# Moreover, this approach is more suitable for objects with small number of instance variables, such as the scikit-learn models, because any addition of new variables requires changes in the save and restore methods.
