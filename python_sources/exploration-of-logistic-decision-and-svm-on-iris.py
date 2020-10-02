
# This kernel is going to  explore data points present in Iris dataset and test the accuracy on
# different classification algorithms..

# Importing Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Import the dataset
dataset = pd.read_csv('../input/IRIS.csv')
dataset.head()
# Here we observe that we don't have any non-empty rows
dataset.describe()
dataset.info()

# Let's check th balancing of every class using seaborn and matplotlib. CLearly all the classes in 
# the target variable are balanced
sns.countplot(dataset.species)

# Let's Visualize the pair plot
g = sns.pairplot(dataset, hue='species')
plt.show()
# Observation: petal_length and petal_width seems to be greatly affecting the classes

#Creating target variable
Y = dataset['species']

# Creating features dataset
X = dataset.iloc[:,2:4]

# Converting the target variable to labels
labelEncoder = LabelEncoder()
Y_encoded    = labelEncoder.fit_transform(Y)

# Splitting the dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y_encoded,random_state = 4,test_size = 0.2)

#Scaling the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Let's test this problem statement with different classifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logit_model = LogisticRegression()
logit_model.fit(X_train,Y_train)
logit_pred = logit_model.predict(X_test)
logit_accuracy = metrics.accuracy_score(Y_test,logit_pred)
print("Accuracy with Logistic Regression is {}".format(logit_accuracy))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train,Y_train)
tree_pred = tree_model.predict(X_test)
tree_accuracy = metrics.accuracy_score(Y_test,tree_pred)
print("Accuracy with Decision Tree is {}".format(tree_accuracy))

# Support Vector Machine
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train,Y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(Y_test,svm_pred)
print("Accuracy with SVM is {}".format(svm_accuracy))

# This accuracy can be improved by tweaking the hyperparameters of the classifier.