#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines Project 
# 
# **Note: This is a set of exercises from the Udemy course, [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/).**
# 
# Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!
# 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data

# In[35]:


import pandas as pd


# In[36]:


iris = pd.read_csv('../input/Iris.csv')
iris.head()


# In[37]:


iris.rename(columns={'SepalLengthCm':'sepal_length', 'SepalWidthCm':'sepal_width', 'PetalLengthCm':'petal_length', 'PetalWidthCm':'petal_width', 'Species':'species'}, inplace=True)


# In[38]:


iris.drop(labels='Id', axis=1, inplace=True)


# Let's visualize the data and get you started!
# 
# ## Exploratory Data Analysis
# 
# Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!
# 
# **Import some libraries you think you'll need.**

# In[39]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**

# In[40]:


sns.pairplot(iris,hue='species',palette='Dark2')


# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

# In[41]:


sns.set_style('whitegrid')
iris_setosa = iris[iris['species']=='Iris-setosa']
sns.kdeplot(iris_setosa['sepal_width'], iris_setosa['sepal_length'],shade=True,cmap='plasma',shade_lowest=False)


# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[44]:


from sklearn.svm import SVC


# In[45]:


model = SVC()


# In[46]:


model.fit(X_train,y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[47]:


preds = model.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef


# In[49]:


print(confusion_matrix(y_test,preds))


# In[50]:


print(classification_report(y_test,preds))


# In[51]:


matthews_corrcoef(y_test,preds)


# Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**

# In[52]:


from sklearn.grid_search import GridSearchCV


# In[53]:


from sklearn.model_selection import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[54]:


pwr = [10.0**(i/3.0) for i in range(-8,9)]
pwr


# In[55]:


param_grid = {'C':pwr, 'gamma':pwr}


# ** Create a GridSearchCV object and fit it to the training data.**

# In[56]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)


# In[57]:


grid.fit(X_train,y_train)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[58]:


grid.best_params_


# In[59]:


preds_grid = grid.predict(X_test)


# In[60]:


print(confusion_matrix(y_test,preds_grid))


# In[61]:


print(classification_report(y_test,preds_grid))


# In[62]:


matthews_corrcoef(y_test,preds_grid)


# You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

# ## Great Job!
