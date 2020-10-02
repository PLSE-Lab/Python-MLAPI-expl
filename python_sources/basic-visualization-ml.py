#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #For Visualization 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")
iris.head()


# Removing ID from iris dataframe

# In[ ]:


iris.drop('Id',inplace=True,axis=1)


# Top 5 records of dataset

# In[ ]:


iris.head()


# Summary of a DataFrame

# In[ ]:


iris.info()


# Descriptive statistics of DataFrame

# In[ ]:


iris.describe()


# Basic Visualization

# 1. Scatter plot

# In[ ]:


sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data= iris ,hue='Species')
plt.title('Sepal Length vs Sepal Width')
plt.show()


# In[ ]:


sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data= iris ,hue='Species')
plt.title('Petal Length vs Petal Width')
plt.show()


# In[ ]:


sns.scatterplot(x = 'PetalLengthCm', y = 'SepalLengthCm', data= iris ,hue='Species')
plt.title('Petal Length vs Sepal Length')
plt.show()


# In[ ]:


sns.scatterplot(x = 'PetalWidthCm', y = 'SepalWidthCm', data= iris ,hue='Species')
plt.title('Petal Width vs Sepal Width')
plt.show()


# Now check the distribution of target column i.e. Species

# In[ ]:


sns.countplot(x = 'Species', data = iris)
plt.show()


# Now check pairwise correlation of columns

# In[ ]:


sns.heatmap(data = iris.corr(),annot=True)
plt.show()


# Voilin plot

# In[ ]:


sns.violinplot(y = 'PetalLengthCm', x = 'Species', data= iris, hue='Species')
plt.show()


# Pairplot

# In[ ]:


sns.pairplot(data= iris, hue='Species',palette='Dark2')


# Data Preprocessing

# In[ ]:


np.unique(iris['Species'])


# Converting target column into numerical values

# In[ ]:


iris['Species'] = pd.Categorical(iris['Species'])
iris['Species'] = iris['Species'].cat.codes.apply(int)
np.unique(iris['Species'])


# In[ ]:


iris.head()


# standardization of features

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(iris.drop('Species',axis=1))
scaled_features = scaler.transform(iris.drop('Species',axis = 1))
iris_feat = pd.DataFrame(scaled_features,columns=iris.columns[:-1])


# Checking top 5 values of new dataframe

# In[ ]:


iris_feat.head()


# Train & Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# Assigning X feature columns & y - Target column

# In[ ]:


X = iris_feat 
y = iris['Species']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Machine Learning

# As Iris Dataset is classification between three types of flowers. I am going to use logistic regression, Decision Tree Classifier, Random Forest Classifier, SVM & Gridsearch, K Neighbors Classifier & DNN classifier with tensorflow estimators.

# 1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log = LogisticRegression(solver='lbfgs',multi_class='auto')


# In[ ]:


log.fit(X_train,y_train)


# In[ ]:


prediction_lr = log.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


print(classification_report(y_test,prediction_lr))
print(confusion_matrix(y_test,prediction_lr))
print('Accuracy score is',accuracy_score(y_test,prediction_lr))


# 2. Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier()


# In[ ]:


dt.fit(X_train,y_train)


# In[ ]:


prediction_dt = dt.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_dt))
print(classification_report(y_test,prediction_dt))
print('Accuracy score is',accuracy_score(y_test,prediction_dt))


# 3. Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=200)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


prediction_rf = rf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_rf))
print(classification_report(y_test,prediction_rf))
print('Accuracy score is',accuracy_score(y_test,prediction_rf))


# 4. SVM & Gridsearch

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


sv = SVC(gamma='auto')


# In[ ]:


sv.fit(X_train,y_train)


# In[ ]:


prediction_sv = sv.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_sv))
print(classification_report(y_test,prediction_sv))
print('Accuracy score is',accuracy_score(y_test,prediction_sv))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,.1,.01,.001,.0001]}


# In[ ]:


gs = GridSearchCV(SVC(),param_grid,verbose=5)


# In[ ]:


gs.fit(X_train,y_train)


# In[ ]:


gs.best_params_ #Checking parameters


# In[ ]:


gs.best_estimator_


# In[ ]:


prediction_gs = gs.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_gs))
print(classification_report(y_test,prediction_gs))
print('Accuracy score is',accuracy_score(y_test,prediction_gs))


# 5. K Neighbors Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


prediction_knn = knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_knn))
print(classification_report(y_test,prediction_knn))
print('Accuracy score is',accuracy_score(y_test,prediction_knn))


# By plotting the graph of error rates determining most accurate value of n_neighbors

# In[ ]:


error_rate =[]

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle = 'dashed', marker = 'o', markerfacecolor = 'red',markersize = 10)
plt.show()


# The lowest error rate is at n_neighbors = 7,8,9,11 & 12 so running K Neighbors Classifier with n_neighbors = 7

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


prediction_knn = knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction_knn))
print(classification_report(y_test,prediction_knn))
print('Accuracy score is',accuracy_score(y_test,prediction_knn))


# 6. DNN classifier with tensorflow estimators

# In[ ]:


import tensorflow as tf


# Generating feature_columns

# In[ ]:


iris_feat.columns #columns of iris_feat 


# In[ ]:


feat_col = []

for col in iris_feat.columns:
    feat_col.append(tf.feature_column.numeric_column(col))


# In[ ]:


feat_col


# In[ ]:


input_fn = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train,batch_size= 10, num_epochs= 10, shuffle=True)


# In[ ]:


classifier =tf.estimator.DNNClassifier(hidden_units=[15,15,15],n_classes=3,feature_columns=feat_col)


# In[ ]:


classifier.train(input_fn,steps= 50)


# In[ ]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x = X_test,batch_size=len(X_test),shuffle=False)


# In[ ]:


prediction_dl =list( classifier.predict(input_fn=pred_fn))


# In[ ]:


prediction_dl


# Need to collect class_ids to check model accuracy with y_test

# In[ ]:


final_pred = []

for pred in prediction_dl:
    final_pred.append(pred['class_ids'][0])


# In[ ]:


final_pred


# In[ ]:


print(confusion_matrix(y_test,final_pred))
print(classification_report(y_test,final_pred))
print('Accuracy score is',accuracy_score(y_test,final_pred))

