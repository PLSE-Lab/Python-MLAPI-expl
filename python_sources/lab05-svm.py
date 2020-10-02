#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine (SVM)
# It's a supervised machine learning algorithm which can be used for both classification or regression problems. But it's usually used for classification. Given 2 or more labeled classes of data, it acts as a discriminative classifier, formally defined by an optimal hyperplane that seperates all the classes. New examples that are then mapped into that same space can then be categorized based on on which side of the gap they fall.
# 
# ![](https://camo.githubusercontent.com/ae3d247a4c7cf5bc9f4134a1a90c0df69b39e988/68747470733a2f2f7777772e64747265672e636f6d2f75706c6f616465642f70616765696d672f53766d4d617267696e322e6a7067)

# In[ ]:


# Simple SVM
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class

plt.plot(xBlue, yBlue, 'ro', color='blue')
plt.plot(xRed, yRed, 'ro', color='red')
plt.plot(2.5,4.5,'ro',color='green')

#
#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far      High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface 
#					High value: all training examples classified correctly but complex surface 
classifier = svm.SVC()
classifier.fit(X,y)

print( classifier.predict([[2.5,4.5]]))

plt.axis([-0.5,10,-0.5,10])

plt.show()


# In[ ]:


# Titanic Dataset
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file_path = "../input/titanicdataset-traincsv/"
data = pd.read_csv(os.path.join(file_path,'train.csv'))

print(data.head())
print(data.describe())
print(data.corr())
print(data.isnull().sum())
data.head()
data.describe()

# replace missing data (Age)
# convert to array
age = data['Age'].values
age = np.reshape(age,(-1,1))
#print(age)

imp = SimpleImputer(missing_values = np.nan , strategy='most_frequent')
imp.fit(age)
data['Age'] = imp.transform(age)
print(data.isnull().sum())

#convert label to int
data.Sex=data.Sex.astype('category').cat.codes
print(data.head())

# input and output data
features = data[["Pclass", "Fare", "Age"]]
target = data.Survived

#features scaling
scale = StandardScaler()
features = scale.fit_transform(features)
print(features[0,:])

# split data for training and testing
feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3, random_state=42)
print(feature_train.shape)
print(feature_test.shape)

#hyperparameter tuning (C)
parameters = { 'C':np.arange(1,11,0.5)}
svc = svm.SVC(gamma='auto')
SVM=GridSearchCV(svc, parameters)
SVM.fit(feature_train,target_train)
print(SVM.best_estimator_)

# prediction
predictions = SVM.predict(feature_test)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))


# ## Checkpoint 1
# From code above, add more features to increase accuracy.

# In[ ]:


# Titanic Dataset
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file_path = "../input/titanicdataset-traincsv/"
data = pd.read_csv(os.path.join(file_path,'train.csv'))

# print(data.head())
# print(data.describe())
# print(data.corr())
# print(data.isnull().sum())
# data.head()
# data.describe()

# replace missing data (Age)
# convert to array
age = data['Age'].values
age = np.reshape(age,(-1,1))
#print(age)

imp = SimpleImputer(missing_values = np.nan , strategy='most_frequent')
imp.fit(age)
data['Age'] = imp.transform(age)
# print(data.isnull().sum())

#convert label to int
data.Sex=data.Sex.astype('category').cat.codes
# print(data.head())

# input and output data
features = data[["Pclass", "Fare", "Age", "Sex", "SibSp", "Parch"]]
target = data.Survived

#features scaling
scale = StandardScaler()
features = scale.fit_transform(features)
# print(features[0,:])

# split data for training and testing
feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3, random_state=42)
print(feature_train.shape)
print(feature_test.shape)

#hyperparameter tuning (C)
parameters = { 'C':np.arange(1,11,0.5)}
svc = svm.SVC(gamma='auto')
SVM=GridSearchCV(svc, parameters)
SVM.fit(feature_train,target_train)
print(SVM.best_estimator_)

# prediction
predictions = SVM.predict(feature_test)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))


# In[ ]:


# Iris dataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

#
#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far      High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface 
#					High value: all training examples classified correctly but complex surface 

dataset = datasets.load_iris()

#print(dataset)

features = dataset.data
targetVariables = dataset.target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.3, random_state=42)

model = svm.SVC(gamma=0.9, C=100000)
#model = svm.SVC()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))


# ## Checkpoint 2
# From code above, adjust the parameters gamma and C to increase accuracy.

# In[ ]:


# Iris dataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

#
#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far      High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface 
#					High value: all training examples classified correctly but complex surface 

dataset = datasets.load_iris()

#print(dataset)

features = dataset.data
targetVariables = dataset.target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.3, random_state=42)

model = svm.SVC(gamma=0.01, C=100)
#model = svm.SVC()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))


# ## Checkpoint 3
# Implement SVM classifier and predict the result using MNIST dataset.

# In[ ]:


# from sklearn import svm
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import mnist
# from keras.utils import to_categorical


# x = mnist.test_images()
# y = mnist.test_labels()

# print("X: {},{}".format(x.shape,x.dtype))
# print("Y: {}".format(y.shape))


# nTrain = x.shape[0]
# nDimTrain = x.shape[1]*x.shape[2]

# x = x.reshape(nTrain,nDimTrain)

# print("# reshape")
# print("X: {}".format(x.shape))
# print("Y: {}".format(y.shape))

# #splitting data
# featureTrain, featureTest, targetTrain, targetTest = train_test_split(x, y, test_size=0.3, random_state=42)

# #training
# model = svm.SVC(gamma=0.01, C=100)
# fittedModel = model.fit(featureTrain, targetTrain)

# #prediction
# predictions = fittedModel.predict(featureTest)

# #result
# print(confusion_matrix(targetTest, predictions))
# print(accuracy_score(targetTest, predictions))


# ## Checkpoint 4
# Implement SVM classifier and predict the result using Cifar10 dataset.
