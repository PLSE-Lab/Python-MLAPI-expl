#Bismillahir Rahmaanir Raheem
#Almadadh Ya Gause RadiAllahu Ta'alah Anh - Ameen
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # for graphical representations of the data
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Import the Breast Cancer Data Set
# 5x Rows and 32x Columns
dataset = pd.read_csv('../input/dataR2.csv')
X = dataset.iloc[:, 1:9].values # independant variables
Y = dataset.iloc[:, 9].values # dependant variable


# Display the first few rows of the data set
dataset.head()

# Dimensions of the Data Set
# I.e. shows the total number of Rows and Columns 
#print(dataset.shape)
print("Dimensions of the Breast Cancer Data Set : {}".format(dataset.shape))
# Note: Number of Rows = 569 and Number of Columns = 32
#'Diagnosis' is the column that will be predicted
# Diagnosis = M (Malignant) and Diagnosis = B (Benign)
# 1 => M and 0 => B 
# Out of 569 Rows, 357 are labelled as B (Benign) and the remaining 212 Rows are M (Malignant)

#Finding any possible missing or null data points - using pandas function
#Na means missing or blank data
dataset.isnull().sum()
dataset.isna().sum()
#Note: there is no missing or null values present in this data set 

#Next, Use Label Encoder to label any categorical data
#Categorical data => any columns that have label values and not numeric values 
#Convert categorical data to numeric data
#print('Printing the values of Y (the predictor variable) BEFORE encoding . . .')
#print(Y)

#Encoding any categorical values
#Changes M and B to 1 and 0, respectively
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#print('Printing the values of Y (the predictor variable) AFTER encoding . . .')
#print(Y)


#Splitting the Data Set into Train and Test
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#The above means, that 25% of the Data Set will be for Testing purposes and the remaining 75% will be used as Training Data Set

#Feature Scaling the Data Set - bringing the features to the same level of magnitudes
#i.e. Transform the Data such that, it fits in a specific scale like 0-100 or 0-1
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)

#Selecting the Model/Machine Learning Algorithm
#Classification Algorithms

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START LOGISTIC REGRESSION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Logistic Regression for the Training Data Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Logistic Regression Model')
print(confusionmatrix)
print()
#59%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END LOGISTIC REGRESSION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START NEAREST NEIGHBOR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#KNeighborsClassifier method from the neighbors class - for Nearest Neighbor Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Nearest Neighbor Model')
print(confusionmatrix)
print()
#72%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END NEAREST NEIGHBOR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#SVC method from SVM class for Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Support Vector Machine Model')
print(confusionmatrix)
print()
#59%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START KERNEL SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#SVC method from SVM class for Kernel SVM Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Kernel Support Vector Machine Model')
print(confusionmatrix)
print()
#66%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END KERNEL SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START NAIVE BAYES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#GaussianNB method from naive_bayes class for Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Naive Bayes Model')
print(confusionmatrix)
print()
#55%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END NAIVE BAYES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START DECISION TREE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#DecisionTreeClassifier from the tree class for Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Decision Tree Model ')
print(confusionmatrix)
print()
#69%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END DECISION TREE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START RANDOM FOREST~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#RandomForestClassifier from the ensemble class for Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
#Predict the Test Results for this Model:
Y_pred = classifier.predict(X_test)
#Check the Accuracy for this Model - use Confusion Matrix as a Metric for evaluating the performance of this Model:
#"Accuracy" = The ratio of the number of correct predictions to the total number of input samples
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix for Random Forest Model ')
print(confusionmatrix)
print()
#48%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END RANDOM FOREST~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#
#RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#1) LOGISTIC REGRESSION - 59% - 4th
#2) NEAREST NEIGHBOR - 72% - 1st
#3) SUPPORT VECTOR MACHINE - 59% - 4th
#4) KERNEL SUPPORT VECTOR MACHINE - 66% - 3rd
#5) NAIVE BAYES - 55% - 5th
#6) DECISION TREE - 69% - 2nd
#7) RANDOM FOREST - 48% - 6th
#















