#!/usr/bin/env python
# coding: utf-8

# Choosing an effective model for predicting Iris species. (Models Covered: Logistic Regression and K Nearest Neighbors)

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris.head()


# In[ ]:


#Create a new column which will act as our target variable
def target(row):
    if row.Species == 'Iris-setosa':
        return 0
    elif row.Species == 'Iris-versicolor':
        return 1
    elif row.Species == 'Iris-virginica':
        return 2
    
iris['species'] = iris.apply(target, axis = 1)    
iris.head()


# In[ ]:


#visualize the relation between different features and species using seaborn
import seaborn as sns
sns.boxplot(x = "Species", y ="SepalLengthCm", data = iris)


# In[ ]:


sns.boxplot(x = "Species", y ="PetalLengthCm", data = iris)


# In[ ]:


sns.boxplot(x = "Species", y ="PetalWidthCm", data = iris)


# In[ ]:


#preparing the feature matrix and target vector for scikit-learn
feature_cols = ['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris[feature_cols]
y = iris['species']


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


#Split the data into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #Logistic regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

#retrain the model with all available data
logreg.fit(X,y)


# #K Nearest Neighbors (KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

#to find the ideal value of k for knn
scores = []
k_range = list(range(1,26))
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Values of K')
plt.ylabel('Accuracy Score')


# From the above plot, it is evident that the accuracy score for KNN peaks at around 0.974 (when k >=5 and k<=16 approximately). The accuracy score for KNN is much higher than that of Logistic Regression (0.92). 
# We thus choose KNN for our model, with K = 6

# In[ ]:


#Retraining our model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)


# 
