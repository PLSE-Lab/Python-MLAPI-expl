#!/usr/bin/env python
# coding: utf-8

# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
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

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')


# In[ ]:


iris.head()


# ## Exploratory Data Analysis

# In[ ]:


sns.pairplot(iris, hue='species')


# In[ ]:


iris_setosa = iris[iris['species'] == 'Iris-setosa']
iris_setosa.head()


# In[ ]:


sns.jointplot(x='sepal_width', y='sepal_length', data=iris_setosa, kind='kde')
plt.show()


# In[ ]:


# Splitting data into training and testing datasets
from sklearn.model_selection import train_test_split


# In[ ]:


X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[ ]:


# Training the model
from sklearn.svm import SVC


# In[ ]:


svm_classifier = SVC(gamma='auto')
svm_classifier.fit(X_train, y_train)


# In[ ]:


# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


y_pred = svm_classifier.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


svm_classifier.score(X_train, y_train)


# ## Gridsearch

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}


# In[ ]:


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_pred = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test, grid_pred))

