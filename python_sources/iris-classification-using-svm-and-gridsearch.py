#!/usr/bin/env python
# coding: utf-8

# # IRIS data set classification using SVM and GridSearch

# Here is the little description about Iris dataset. 
# 
# The data set consists of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# Setosa flower

# ![](http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg)

# Versicolor flower

# ![](http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg)

# Virginica flower

# ![](http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg)

# ### What is SVM ?
# SVM is a powerful and flexible class of supervised algorithms for both classification and regression.
# Support Vectors are the simple co-oridnates of individual obeservation. 
# Support Vecrtor Machine(SVM) is a borderline/boundary/limit which segregates the features.
# 
# ### How to draw the line ?
# The simplest way to interpret the objective function in a SVM is to find the minimum distance of the frontier from the closest support vector. Once we have these distances for all the frontiers, we simply choose the frontier with the maximum distance(i.e from the closest support vector ).
# 
# For simple cases it is easy to find a straight line to classify the features. But in some cases, we need to map a vector to a higher dimension plane so that they get segregated from each other. 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# Let us import the Iris data set

# In[ ]:


data = pd.read_csv("../input/Iris.csv")
data.head()


# In[ ]:


data.info()


# Let us create a pairplot of the data set

# In[ ]:


sns.pairplot(data, hue = 'Species');


# Create a kde plot of sepal length versus sepal width for versicolor flower

# In[ ]:


versicolor = data[data['Species'] == 'Iris-versicolor']
sns.kdeplot( versicolor['SepalWidthCm'], versicolor['SepalLengthCm'], 
            cmap='plasma', shade=True, shade_lowest=False)


# Split the data as Train and Test sets

# In[ ]:


from sklearn.model_selection import train_test_split
X = data.drop('Species', axis = 1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# It's time to train the model 

# In[ ]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)


# Let us do the predictions using the trained model

# In[ ]:


pred = svm.predict(X_test)


# Create confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))


# Create classification report

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


svm.score(X_test,y_test)


# ###  Grid Search
# Now we will tune the parameters, check for the improvement
# 
# Tuning parameters value for machine learning algorithms effectively improves the model performance ( Let us check in for our data)

# Import Gridsearch from Scikit Learn.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# Create a dictionary called param_grid and fill out some parameters for C and Gamma

# In[ ]:


param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}


# Create a GridSearchCV object and fit it to the training data
# 
# Grid search is a model hyperparameter optimization technique.

# In[ ]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
grid.fit(X_train, y_train)


# Let us predict using the Grid model

# In[ ]:


pred_grid = grid.predict(X_test)


# Let us compute the confusion matrix

# In[ ]:


print(confusion_matrix(y_test, pred_grid))


# Let us print the report also

# In[ ]:


print(classification_report(y_test, pred_grid))


# The predictions looks the same in this case. 
# If any noise in the data, we need to tune the parameters, but it should not overfit the model.
