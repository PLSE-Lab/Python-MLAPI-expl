#!/usr/bin/env python
# coding: utf-8

#  In this problem we have use 30 columns and we have to predict the stage of breast cancer M for Malignant and B for Benign.
# 
# 

# 1.    This analysis hs been done using KNN and SVM With detailed explanation.
# 

# 1. Lets Start.

# > Attribute Information:
# 
# 1) ID number
# 
# 2) Diagnosis (M = malignant, B = benign)
# 
# -3-32.Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g). concavity (severity of concave portions of the contour)
# 
# h). concave points (number of concave portions of the contour)
# 
# i). symmetry
# 
# j). fractal dimension ("coastline approximation" - 1)
# 
# 5 here 3- 32 are divided into three parts first is Mean (3-13), Stranded Error(13-23) and Worst(23-32) and each contain 10 parameter (radius, texture,area, perimeter, smoothness,compactness,concavity,concave points,symmetry and fractal dimension)
# 
# Here Mean means the means of the all cells, standard Error of all cell and worst means the worst cell
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


df.head()


# **Data Pre-processing
# **

# In[ ]:


df.info()


# In[ ]:


#there are a 33 columns and 569 entries


#  float have 31 column, int have 64 and object has only 1 column

# Now we can see Unnamed:32 have 0 non null object it means the all values are null in this column so we cannot use this column for our analysis*
# 
# There is an id,that can not use for KNN.

# In[ ]:


y = df.diagnosis
list = ['Unnamed: 32','id','diagnosis']
X = df.drop(list,axis=1) # drop unnamed: 32 column


# In[ ]:


df.head(2)


# **Visualization**

# In[ ]:


#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In this dataset firstly fing the relationship between 'no of Benign',and 'no of malign' by using countplot
# 

# In[ ]:


sns.countplot(df['diagnosis'],label='Count')


# barplot easily says that,there are more number of 'Benign' is present,but we dont know how much? lets find it

# In[ ]:


B,M = y.value_counts()
print('Number of Benign:',B)
print('Number of Malignant:',M)


# there are 357 'Benign',and 212 'Malign' are present.

# In[ ]:


#lets draw correlation graph , we use correlation because it use to remove multi colinearity it means the column are depending on each other so we should avoid it because it use same column twice.


# In[ ]:


corr = df[df.columns[1:11]].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar ='True',annot=True,linewidths=.5,fmt='.1f',cmap='coolwarm')

#corr = data[features_mean].corr() # .corr is used for find corelation
#plt.figure(figsize=(14,14))
#sns.heatmap(corr, cbar = 'True',square = True, annot='True', fmt= '.2f',annot_kws={'size': 15},
 #          xticklabels= features_mean, yticklabels= features_mean,
  #         cmap= 'coolwarm') # for more on heatmap


# to chose the feature by using heatmap

# In[ ]:


X.head()


# # 1) KNN

# In[ ]:


#import the libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X = StandardScaler().fit_transform(X.values)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5,
                           p=2, metric='minkowski')


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))        


# In[ ]:


print_score(knn, X_train, y_train, X_test, y_test, train=True)


# our model fit in train database as 98% accuracy.
# 
# and 286(class 0) has malignant, and 160(class 1) has benign. only 9 value that not converted any class.

# In[ ]:


print_score(knn, X_train, y_train, X_test, y_test, train=False)


# our model fit in test  database as 94% accuracy.
# 
# and 68(class 0) has malignant, and 40(class 1) has benign. only 6 value that not converted any class.

# **our task is convertion of all value to any class, now we apply SVM**

# # 2) Support Vector Machin

# In[ ]:


from sklearn.svm import SVC #import library


# In[ ]:


model = SVC()


# In[ ]:


model.fit(X_train,y_train)  #fit the model


# In[ ]:


y_pred = model.predict(X_test)  #prediction


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


# our model fit in test and train database as 97% accuracy.
# 
# and 70(class 0) has malignant, and 41(class 1) has benign. only 3 value that not converted any class.

# # 3)Linear SVC 

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train,y_train)
y_pred = linear_svc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


# linear svc gives worst result as compare to svc.

# # 4)GridSearchCV

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV 


# In[ ]:


# C -> controls the cost of the misclassification on the training data.
# A large C value gives you the low bias and high variance
# Lower C value gives you the high bias and the lower variance.
# gamma is a free parameter in radial basis function.
# Higher gamma value leads to Higher bias and lower variance value.
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,verbose=3) # put the verbose = 3


# In[ ]:


grid.fit(X_train,y_train)  #assignment: scalling the X_strain


# In[ ]:


grid.best_params_  #we find best parameter


# In[ ]:


grid.best_estimator_  #find best estimator


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, grid_predictions))           #does not identify six value correctly 


# now we convert 71 to malignant and 41 to benign.

# In[ ]:


print(classification_report(y_test, grid_predictions))


# model has 98% accuracy.
