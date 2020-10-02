#!/usr/bin/env python
# coding: utf-8

# # KNN Project
# ## Tracy(11/15/2019)_Iris Species
# Use Python to visualize machine learning algorithms to classify the Iris dataset.

# ## Import Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (10,6)


# ## Content:
# * EDA
# * Separating Features and Labels
# * Converting String Value To int Type for Labels
# * Data Standardisation
# * Splitting Dataset into Training Set and Testing Set
# * Build KNN Model with Default Hyperparameter
# * Accuracy Score
# * Confusion Matrix with Seaborn - Heatmap

# In[ ]:


## Import Data
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


# In[ ]:


iris.head()


# In[ ]:


iris.sample(5)


# In[ ]:


iris.info()


# ## EDA

# In[ ]:


sns.pairplot(iris, hue='PetalWidthCm',palette='coolwarm')


# ## Separating Features and Labels

# In[ ]:


X = iris.iloc[:, 1:5]
X.head()


# ## Converting String Value to Int Types
# * Iris-setosa -> 0
# * Iris-versicolor -> 1
# * Iris-virginica -> 2

# In[ ]:


iris.Species.unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
y = iris.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)


# ## Data Standardisation

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# ## Splitting Dataset into Training Set and Testing Set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# ## Building KNN Model with Dafault Hyperparameter

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


# ## Accuracy Score

# In[ ]:


print('Accuracy Score:')
print(metrics.accuracy_score(y_test, predictions))


# In[ ]:




