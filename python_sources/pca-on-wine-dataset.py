#!/usr/bin/env python
# coding: utf-8

# Hello everyone!!!
# 
# Today i am working on PCA (Principal component analysis).
# If you like then please vote me!!

# # Introduction 
# 
# Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
#              Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

# # Steps to do it
# 
# * Step 1: Standardization
# * Step 2: Covariance Matrix computation
# * Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
# * Step 4: Feature vector
# * Step 5: Recast the data along the principal components axes

# # For brief information please visit below link:
# 
# https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-b836fb9c97e2

# # Please check below plot for more information...
# This is how actually it works.....
# 
# 
# ![](data:image/png;base64,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)

# # Exercise

# In[ ]:


#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))


# In[ ]:


#Reading the dataset..
df=pd.read_csv("../input/wineQualityReds.csv")


# In[ ]:


#Checking the starting "5" values.
df.head()


# In[ ]:


#Checking if there is any existing null value or not
df.isnull().sum()


# In[ ]:


#Checking the unique values from "quality column"
df["quality"].unique()


# In[ ]:


#Count the unique values in "quality column"
df["quality"].value_counts()


# In[ ]:


#Plot for quality
df["quality"].value_counts().plot.bar(color='Yellow')
plt.xlabel("Quality score")
plt.legend()


# In[ ]:


#Checking the dimensions
df.shape


# In[ ]:



#Separating dependent and independent variable.
X = df.iloc[:, 1:12].values
y = df.iloc[:, 12].values



# In[ ]:


print(X)


# In[ ]:


print(y)


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:



# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[ ]:


print(explained_variance)


# In[ ]:


#Fitting Logistic regression into dataset
from sklearn.linear_model import LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
print("The accuracy of  LogisticRegression is:",accuracy_score(y_test, lr_pred))


# In[ ]:


print(lr_cm)


# In[ ]:


#Fitting Randomforest into dataset
from sklearn.ensemble import RandomForestClassifier
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
print("The accuracy of RandomForestClassifier is:",accuracy_score(rdf_pred,y_test))


# In[ ]:


print(rdf_cm)


# In[ ]:


#Fitting KNN into dataset
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
knn_cm=confusion_matrix(y_test,knn_pred)
print("The accuracy of KNeighborsClassifier is:",accuracy_score(knn_pred,y_test))


# In[ ]:


print(knn_cm)


# In[ ]:


#Fitting Naive bayes into dataset
from sklearn.naive_bayes import GaussianNB



gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
bayes_pred=gaussian.predict(X_test)
bayes_cm=confusion_matrix(y_test,bayes_pred)
print("The accuracy of naives bayes is:",accuracy_score(bayes_pred,y_test))


# In[ ]:


print(bayes_cm)


# # Conclusion
# 
#      Here my intension is to apply PCA on wine dataset ,i am not bother about the accuracy right now....However you can solve it by checking its correlation(using heatmap or df,corr()) between each variable and remove the variables which are less correlated to your dependent variable......For sure you will get good accuracy...
#      