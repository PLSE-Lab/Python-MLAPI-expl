#!/usr/bin/env python
# coding: utf-8

# 
# # Support Vector Machines Project 
# 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[ ]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


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
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

# In[ ]:


#This code doest not work in kaggle 
# import seaborn as sns
# iris=sns.load_dataset('iris')
#follow this
import pandas as pd
iris=pd.read_csv("../input/Iris.csv")


# Let's visualize the data and get you started!
# 
# ## Exploratory Data Analysis
# 
# Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!
# 
# **Import some libraries you think you'll need.**

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**

# In[ ]:


sns.pairplot(data=iris,hue='Species')


# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

# In[ ]:


setosa = iris[iris['Species']=='Iris-setosa']
sns.kdeplot( setosa['SepalWidthCm'], setosa['SepalLengthCm'],
                 cmap="plasma", shade=True, shade_lowest=False)


# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = iris.drop('Species',axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC()


# In[ ]:


svc_model.fit(X_train,y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[ ]:


predict=svc_model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,predict))


# In[ ]:


print(classification_report(y_test,predict))


# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**

# In[ ]:


from sklearn.grid_search import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# ** Create a GridSearchCV object and fit it to the training data.**

# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,grid_predictions))


# In[ ]:


print(classification_report(y_test,grid_predictions))

