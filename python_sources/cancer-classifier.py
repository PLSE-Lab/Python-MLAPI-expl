#!/usr/bin/env python
# coding: utf-8

# # Data information:
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# # First we import the necessary libraries and functions

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# # Now we read in the data file

# In[ ]:


data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head()


# # We then split the features from the labels
# ## We also drop a row full of NaNs

# 

# In[ ]:


X = data.drop(columns = ['Unnamed: 32','diagnosis', 'id'])
labels = data[['diagnosis']]


# # The original labels are strings, we must convert them to boolean values

# In[ ]:


y = labels.apply(lambda x: x=='M')
y.head(20)


# # We split the data into training, validation, and testing sets

# In[ ]:


X_train,X_eval,y_train, y_eval = train_test_split(X,y)
X_eval, X_test, y_eval, y_test = train_test_split(X_eval,y_eval, test_size = .5)


# # We initialize the model

# In[ ]:


model = XGBClassifier(n_estimators = 1000, max_depth = 3)


# ## We know train the model using our training and validations sets 

# In[ ]:


model.fit(X_train, y_train, eval_set =[(X_eval, y_eval)], early_stopping_rounds = 100, verbose = 10)


# # We predict labels for our test set and compare them to the actual labels
# 

# In[ ]:


test_predictions = model.predict(X_test)


# # Now we measure model accuracy

# In[ ]:


score = accuracy_score(y_test, test_predictions)
print( 'Accuracy: ', score)


# # The model is able to distinguish between patients with bening and malignant cancers with a 97.22% accuracy!!
