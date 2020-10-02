#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Problem-Statement" data-toc-modified-id="Problem-Statement-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Problem Statement</a></span></li><li><span><a href="#Loading-Packages-and-Data" data-toc-modified-id="Loading-Packages-and-Data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Loading Packages and Data</a></span></li><li><span><a href="#Data-Structure-and-Content" data-toc-modified-id="Data-Structure-and-Content-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Structure and Content</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Univariate-Analysis" data-toc-modified-id="Univariate-Analysis-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Univariate Analysis</a></span></li><li><span><a href="#Bivariate-Analysis" data-toc-modified-id="Bivariate-Analysis-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Bivariate Analysis</a></span></li></ul></li><li><span><a href="#Spliting-Data-into-train-and-test" data-toc-modified-id="Spliting-Data-into-train-and-test-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Spliting Data into train and test</a></span></li><li><span><a href="#Preprocessing-Data" data-toc-modified-id="Preprocessing-Data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Preprocessing Data</a></span><ul class="toc-item"><li><span><a href="#Feature-Scaling" data-toc-modified-id="Feature-Scaling-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Feature Scaling</a></span></li></ul></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Modeling</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li></ul></li></ul></div>

# # Introduction[](#Introduction)
# 
# Iris dataset is said to be the the first step towards Machine Learning and through this dataset, I will try to elloborate different aspects of machine learning. The Iris dataset was used in R.A. Fisher's classic 1936 paper, and is one of the most prominant datasets for classification problem.

# # Problem Statement[](#Problem-Statement)
# 
# For every Machine Learning problem, the first step is to be defining the problem statement. Here the problem is already well defined here i.e. to classify the species of leaf.

# # Loading Packages and Data[](#Loading-Packages-and-Data)

# In[ ]:


import numpy as np 
import pandas as pd #For loading dataset
import matplotlib.pyplot as plt # Visualization Library
import seaborn as sns # Visualization Library


# In[ ]:


import os
print(os.listdir("../input"))
dataset = pd.read_csv('../input/Iris.csv')


# # Data Structure and Content[](#Data-Structure-and-Content)
# 
# Let's check out the data

# In[ ]:


dataset.info()


# This shows that our data consist of 150 samples and 6 columns. All the independent variables are numeric and dependent variable is categorical.

# In[ ]:


dataset.head()


# Here we have given Sepal and Petal length and width using which we have to classsify the species of the leaf.

# # Exploratory Data Analysis[](#Exploratory-Data-Analysis)
# 
# ## Univariate Analysis[](#Univariate-Analysis)
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x=dataset['Species'],data=dataset)


# First let's check out our dependent variable. The plot shows that each category has same no. of samples.

# In[ ]:


sns.distplot(dataset['SepalLengthCm'], kde = False)


# In[ ]:


sns.distplot(dataset['SepalWidthCm'], kde = False)


# In[ ]:


sns.distplot(dataset['PetalLengthCm'], kde = False)


# In[ ]:


sns.distplot(dataset['PetalWidthCm'], kde = False)


# ## Bivariate Analysis[]()

# In[ ]:


sns.set_style("ticks")
sns.pairplot(dataset,hue = 'Species',diag_kind = "kde",kind = "scatter",palette = "husl")


# Let's check the graph more clearly

# In[ ]:


sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=dataset, fit_reg=False,hue='Species') 


# In[ ]:


sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', data=dataset, fit_reg=False,hue='Species') 


# We can see through the graph that we can classify Iris-Setosa with petal or sepal width and length. But its difficult to find the other two without modeling.

# # Spliting Data into train and test[](#Spliting-Data-into-train-and-test)

# In[ ]:


#Seprating dependent and independent variable
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

#Split dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.25,random_state=0)


# # Preprocessing Data[](#Preprocessing-Data)
# 
# ## Feature Scaling[](#Feature-Scaling)

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# # Modeling[](#Modeling)
# 
# ## Logistic Regression[](#Logistic-Regression)

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# Predict the test set results

# In[ ]:


y_pred = classifier.predict(X_test)


# Finding out the accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# So we got 81% accuracy with logistic regression. Let's find out other models accuracy.

# In the upcoming versions, I will focus more on data viz explanation and classifying using other models like Decision Tree, Random Forest, XGBoost and ANN.
