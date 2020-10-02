#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Contents
# 
# * [<font size=4>Getting Started</font>](#1)
#     * [Importing the Libraries](#1.1)
#     * [Importing and Inspecting the Data](#1.2)
#    
#    
# * [<font size=4>Fitting the model</font>](#2)
#     * [Setting up the input and the output variable](#2.1)
#     * [Fitting The Naive Bayes Model](#2.2)
#     * [Inspecting the Model](#2.3)
#     * [Some other Trivia](#2.4)
#     

# In[ ]:


# Getting Started <a id="1.1"></a>
Here we describe importing the library, impoting the datset and some basic checks on the dataset


# # Import Libraries <a id="1.1"></a>

# Male  0 
# Female 1

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
# Import LabelEncoder
from sklearn import preprocessing


# # Importing and Inspecting the Data <a id="1.2"></a>

# In[ ]:


# Imporitng the File
nbflu=pd.read_csv('/kaggle/input/naivebayes.csv')
# Checking the shape of the data frame
print(nbflu.shape)
# Printig top few rows
print(nbflu.head(5))


# # Fitting the model <a id="2"></a>
# Here we describe setting the input, out variable and fitting of the model

# # Setting up the input and the output variable <a id="2.1"></a>

# In[ ]:


# Collecting the features and target individually
x1= nbflu.iloc[:,0]
x2= nbflu.iloc[:,1]
x3= nbflu.iloc[:,2]
x4= nbflu.iloc[:,3]
y=nbflu.iloc[:,4]
list(nbflu.index[:4])


# In[ ]:


#creating labelEncoder
le = preprocessing.LabelEncoder()
x1= le.fit_transform(x1)
x2= le.fit_transform(x2)
x3= le.fit_transform(x3)
x4= le.fit_transform(x4)
y=le.fit_transform(y)

X = pd.DataFrame(list(zip(x1,x2,x3,x4)))
X


# # Fitting The Naive Bayes Model <a id="2.2"></a>

# class sklearn.naive_bayes.CategoricalNB(alpha=1.0, fit_prior=True, class_prior=None)[source]

# In[ ]:


#Create a Gaussian Classifier
model = CategoricalNB()

# Train the model using the training sets
model.fit(X,y)

#Predict Output
#['Y','N','Mild','Y']
predicted = model.predict([[1,0,0,1]]) 
print("Predicted Value:",model.predict([[1,0,0,1]]))
print(model.predict_proba([[1,0,0,1]]))


# # Looking at the model <a id="2.3"></a>

# In[ ]:


# Looking at the model parameters
print(model.get_params())
# Checking the likelyhood Table
print(model.category_count_[0])


# # Some other Trivia <a id="2.2"></a> 

# Other varieties
# * Gaussian Naive Bayes :  implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian
# * MultinomialNB implements the naive Bayes algorithm for multinomially distributed data,
# * Partial Fit. Scaling with instances using out-of-core learnig

# * Let us predict Fever = No, Chills = N, Headache = N, runnynose = N
#      Fever = yes, Chills = Yes, Headache = N, runnynose = N
#       Fever = yes, Chills = Yes, Headache = Strong, runnynose = N
# * Alpha = 2 See the prediction
# * Make prior as false
