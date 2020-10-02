# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/heart.csv"))
import os
print(os.listdir("../input"))
dataset= pd.read_csv('../input/heart.csv')

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:21:13 2019

@author: Abhijit Biswas
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
#dataset= pd.read_csv("../input/heart.csv")
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:,-1].values

# Checking the presence of null values
dataset.isnull().sum()

""" No Categorical Values Present """

# Splitting the datasets
from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y , test_size=0.25, random_state = 0)

# Implying the standard scaler
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
X_train= standardscaler.fit_transform(X_train)
X_test= standardscaler.transform(X_test)

""" Need to apply the Classification Problem """

"""Logistic Regression"""
from sklearn.linear_model import LogisticRegression
LR_regressor= LogisticRegression()
LR_regressor.fit(X_train, Y_train)

Y_pred_LR = LR_regressor.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(Y_test, Y_pred_LR)
cm_lr
# We got accuracy as 83 % --((24+39)/(24+39+4+9)

# K Cross Validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= LR_regressor , X=  X_train, y= Y_train, cv = 10)
accuracies.mean() #82.8
accuracies.std() #4.5

""" Decision Tree """
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier()
DT_classifier.fit(X_train, Y_train)

Y_pred_DT = DT_classifier.predict(X_test)

# Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_DT= confusion_matrix(Y_test, Y_pred_DT)

#accuracy= 78.9 % ((25+35)/(25+35+16)

# creating the K Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_DT = cross_val_score(estimator= DT_classifier, X= X_train, y= Y_train, cv= 10)
accuracies_DT.mean()# 75.411
accuracies_DT.std() #7.3

""" Random Forest """
from sklearn.ensemble import RandomForestClassifier
RF_classifier= RandomForestClassifier(n_estimators = 300)
RF_classifier.fit(X_train, Y_train)

Y_pred_RF = RF_classifier.predict(X_test)
 
from sklearn.metrics import confusion_matrix
cm_RF =confusion_matrix(Y_test, Y_pred_RF)
cm_RF
# (34+39)/(34+39+13) --84.88

# using k cross validation
from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator= RF_classifier, X= X_train, y = Y_train, cv= 10)
accuracies_RF.mean() #82.4
accuracies_RF.std()# 05.0

""" Using K Nearest Neighbors """
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier= KNeighborsClassifier()
KNN_classifier.fit(X_train, Y_train)

Y_pred_KNN= KNN_classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN= confusion_matrix(Y_test, Y_pred_KNN)
cm_KNN
#(23+39)/(23+39+14) --81.5

# importing the k cross validation
from sklearn.model_selection import cross_val_score
accuracies_KNN= cross_val_score(estimator= KNN_classifier, X= X_train, y= Y_train)
accuracies_KNN.mean() #81
accuracies_KNN.std() #3

""" Naive Bayes """
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, Y_train)

Y_pred_NB= NB_classifier.predict(X_test)

# importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(Y_test, Y_pred_NB)
cm_NB
(24+39)/(24+39+13) #82.9

# importing the k cross validation
from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator= NB_classifier, X= X_train, y = Y_train, cv= 10)
accuracies_NB.mean() #82
accuracies_NB.std() #7

""" SVM """
from sklearn.svm import SVC
SVC_classifier = SVC(kernel='linear')
SVC_classifier.fit(X_train, Y_train)

Y_pred_SVC= SVC_classifier.predict(X_test)

# importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm_SVC= confusion_matrix(Y_test, Y_pred_SVC)
cm_SVC 
# (24+41)/(24+41+11)  --85.5

# importing the k cross validation
from sklearn.model_selection import cross_val_score
accuracies_SVC= cross_val_score(estimator = SVC_classifier, X= X_train, y= Y_train)
accuracies.mean() #82.83
accuracies.std()#4.56

""" SVC - non linear """
from sklearn.svm import SVC
SVC_classifier_rbf = SVC()
SVC_classifier_rbf.fit(X_train, Y_train)

Y_pred_SVC_rbf= SVC_classifier.predict(X_test)

# Importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm_SVC_rbf= confusion_matrix(Y_test, Y_pred_SVC_rbf)
cm_SVC_rbf
#(24+41)/(24+41+9+2)--85.5

# Using k cross validation
from sklearn.model_selection import cross_val_score
accuracies_SVC_rbf= cross_val_score(estimator= SVC_classifier_rbf, X= X_train, y= Y_train)
accuracies_SVC_rbf.mean() #81.51
accuracies_SVC_rbf.std() #4.62

""" NOW APLYING GRID ON THE BASIS OF SVM """
from sklearn.model_selection import GridSearchCV
parameters= [{'C':[0.1, 1,10,100,1000], 'kernel':['linear']},
             {'C':[0.1, 1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1,0.5,1,5, 8,10, 50, 100]}]
grid_search= GridSearchCV(estimator= SVC_classifier_rbf,
                          param_grid= parameters,
                          scoring= 'accuracy',
                          n_jobs= -1,
                          cv= 10)
grid_search=grid_search.fit(X_train, Y_train)
SVC_classifier_rbf.get_params().keys()
grid_search_accuracy= grid_search.best_score_ #82
grid_search_parameters = grid_search.best_params_ #linear

""" We shall be applying be applying the Linear model only """

# Feature extraction 
from sklearn.decomposition  import PCA
pca = PCA(n_components = 1)
X_train= pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance= pca.explained_variance_ratio_
# 7 features covering 80%

''' Now applying the logistic regression '''
from sklearn.linear_model import LogisticRegression
LR_PCA_regressor = LogisticRegression()
LR_PCA_regressor.fit(X_train, Y_train)

Y_pred_LR_PCA = LR_PCA_regressor.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= LR_PCA_regressor, X= X_train, y = Y_train, cv = 10, n_jobs=-1)
accuracies.mean() #82.8
accuracies.std() #4.5

"""Now applying the kernel SVM classifier """
from sklearn.svm import SVC
SVC_PCA_regressor = SVC(kernel = 'linear')
SVC_PCA_regressor.fit(X_train, Y_train)

Y_pred_SVC_PCA = SVC_PCA_regressor.predict(X_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= SVC_PCA_regressor, X= X_train, y= Y_train)
accuracies.mean() #83.2
accuracies.std() # 4.3