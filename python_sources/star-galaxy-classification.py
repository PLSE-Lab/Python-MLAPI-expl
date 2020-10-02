#!/usr/bin/env python
# coding: utf-8

#  # Star - Galaxy Classification
#  
#  We are going to classify "SkyServer" data for separation of star or galaxy. For this classification we'll use different classification algorithms for performance measurement of prediction percentage. The following algorithms to be used are :
#  
# * [Data Exploration (EDA)](#7)
#     * [Data Visualization](#13)
# * [Logistic Regression](#1)
# * [KNN (K-Nearest Neighbors Algorithm)](#2)
#     * [Hyperparameter tuning](#12)
# * [Cross Validation](#11)
# * [SVN (Support Vector Machines)](#3)
# * [Naive Bayes Classification](#4)
# * [Decision Tree Classification](#5)
# * [Random Forest Classification](#6)
# * [Confusion Matrix - Classification of Model Evaluation](#9)<br/>
#     * In addition to these, the prediction results of above algorithms will be evaluate by using confusion matrix.
# * [Conclusion](#10)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")


# <a id="7"></a> 
# ***Data Exploration (EDA)***
# * Data Preparation and Formatting

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# <a id="13"></a> 
# **Data Visualization**
# 
# * u = better of DeV/Exp magnitude fit
# * g = better of DeV/Exp magnitude fit
# * r = better of DeV/Exp magnitude fit
# * i = better of DeV/Exp magnitude fit
# * z = better of DeV/Exp magnitude fit
# 
# The Thuan-Gunn astronomic magnitude system. u, g, r, i, z represent the response of the 5 bands of the telescope.

# In[ ]:


# We don't need objid, specobjid, rerun for classification
data.drop(["objid", "specobjid", "rerun"], axis = 1, inplace = True)


# In[ ]:


# QSO data deleting for binary classification. We just need star and galacy classes
data = data[data["class"] != "QSO"]


# In[ ]:


sns.countplot(x= "class", data = data)
data["class"].value_counts()


# We have 4998 Galaxies and 4152 stars. Lets look their astronomic magnitude values by classes.

# In[ ]:


sns.pairplot(data.loc[:,["u", "g", "r", "i", "z", "class"]], hue = "class")
plt.show()


# In[ ]:


# Galaxy = 1 and Star = 0
data['class_binary'] = [1 if i == 'GALAXY' else 0 for i in data.loc[:,'class']]


# In[ ]:


# Convert STAR and GALAXY classes to int. For binary classification
data["class"] = [1 if each == "GALAXY" else 0 for each in data["class"]] 
# After converting operation. We call Star as 0 and Galaxy as 1


# In[ ]:


# data after preparation - formatting operations
data.head()


# In[ ]:


# value selection and normalization
y = data["class"].values
x_data = data.drop(["class"], axis = 1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))


# In[ ]:


# after normalization
x.head()


# In[ ]:


# data separation for train - test operations
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


# Dictionary for score results of classification models
algorithmPerfomanceDict = {}
#algorithmPerfomanceDict = {'ClassificationModel': 1, 'Accuracy': 2}


# <a id="1"></a> 
# **Logistic Regression**<br/>
# * Binary Classification Model (0-1)
# * It has two results 0 and 1.
# * Simplest neural network.
# * Weights ara calculated with forward and backward propogation processes.
# 

# ![BinaryClassification.png](http://i65.tinypic.com/28moies.png)

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
logisticRegressionScore = lr.score(x_test, y_test)
print("Score of Logistic Regression : {0}".format(logisticRegressionScore))
algorithmPerfomanceDict['LogisticRegression'] = logisticRegressionScore


# In[ ]:


# ROC Curve with logistic regression
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report

x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# In[ ]:


data.drop(["class_binary"], axis = 1, inplace = True)


# <a id="2"></a> 
# ***KNN (K-Nearest Neighbors Algorithm)***<br/>
# * An object is classified by a majority vote of its neighbors. 
# * Choose s "k" value.
# * Find the closest point to "k" value.  Euclidean distance is used to find the closest point to k.
# 
# ![EuclideanDistance.png](http://i67.tinypic.com/sv1mcj.png)

# * Calculate the neighbor class count
# * Calculate mojority vote of its neighbors.
# 
# ![KNN_NearestNeighbor.png](http://i68.tinypic.com/2itllk6.png)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1) # Number of neighbors to consider.
knn.fit(x_train, y_train)
knnScore = knn.score(x_test, y_test)
print("Score of KNN Regression : {0}".format(knnScore))
algorithmPerfomanceDict['KNeighborsClassifier'] = knnScore


# In[ ]:


#Lets find best K value
scoreList = []
for each in range(1, 20):
    optimumKnn = KNeighborsClassifier(n_neighbors = each)
    optimumKnn.fit(x_train, y_train)
    scoreList.append(optimumKnn.score(x_test, y_test))
    
plt.plot(range(1, 20), scoreList)
plt.xlabel("K value")
plt.ylabel("Score - Accuracy")
plt.show();


# <a id="12"></a> 
# **Hyperparameter tuning**
# * Another way to find best k value

# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 3)
knn_cv.fit(x, y)
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# <a id="11"></a> 
# ***Cross Validation***
# * Classification algorithms can produce different results at the different random state. Below is the cross validation measures its R^2 score and gives us an average score.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, x, y, cv = k) # uses R^2 as score 
print('CV Scores : ',cv_result)
print('CV Average Score : ',np.sum(cv_result) / k)


# In[ ]:


# According to plot best value for KNN algorithm is 1. It has highest accuracy percentage.


# <a id="3"></a> 
# ***SVM (Support Vector Machines)***<br/>
# It finds best decision boundry between given training data. A support vector machine makes the margin value max. Margin is the distance of two lines which divide the data. The dashed line of below chart is SVM.
# 
# ![SVM.png](http://i65.tinypic.com/2qn0dc9.png)
# 

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state = 42)
svm.fit(x_train, y_train)
svmScore = svm.score(x_test, y_test)
print("Accuracy of Support Vector Machine is : ", svmScore)
algorithmPerfomanceDict['SVM'] = svmScore


# <a id="4"></a> 
# ***Naive Bayes Classification***<br/>
# 
# It calculates probability of factors and create similarity ranges than selects the result with the highest probability. 
# 
# ![NaiveBayes.PNG](http://i64.tinypic.com/xc68oj.png)
# 
# **P(A|B)** : Probability of A given B<br/>
# **P(B|A)** : Probability of B given A<br/>
# **P(A)** : Probability of A<br/>
# **P(B)** : Probability of B<br/>
# 
# 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
nb.score(x_test, y_test)
naiveBayesScore = nb.score(x_test, y_test)
print("Accuracy of Naive Bayes Classifier is : ", naiveBayesScore)
algorithmPerfomanceDict['NaiveBayesClassifier'] = naiveBayesScore


# <a id="5"></a> 
# ***Decision Tree Classification***<br/>
# 
# Decision Tree splits data for classification. This process is performed until each class becomes pure. Pure means each class has own area that contains onlu 1 type of class.
# 
# ![DecisionTree.PNG](http://i65.tinypic.com/2jbs29j.png)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
decisionTreeScore = dt.score(x_test, y_test)
print("Accuracy of Decision Tree Classifier is : ", decisionTreeScore)
algorithmPerfomanceDict['DecisionTreeClassifier'] = decisionTreeScore


# <a id="6"></a> 
# ***Random Forest Classification***<br/>
# 
# Random Forest Classifier is an ensamble learning model.It creates decision tree sets from randomly selected subsets of train dataset and it determines results of the decision trees for the final prediction of test object.
# 
# ![RandomForest.PNG](http://i66.tinypic.com/29zd9n4.png)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42) #Number of trees in forest.
rf.fit(x_train, y_train)
randomForestScore = rf.score(x_test, y_test)
print("Accuracy of Random Forest Classifier is : ", randomForestScore)
algorithmPerfomanceDict['RandomForestClassifier'] = randomForestScore


# <a id="8"></a> 
# ***Comparison of Classifiers***

# In[ ]:


algorithmPerfomanceDict


# In[ ]:


comparisonData = pd.DataFrame.from_dict(algorithmPerfomanceDict, orient = 'index', columns = ["Accuracy"])
comparisonData.head(10)


# In[ ]:


plt.figure(figsize = (20, 7))
sns.barplot(x = comparisonData.index, y = comparisonData.Accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Classification Model')
plt.title('Accuracy Values of Classification Models', color = 'blue', fontsize = 15)
plt.show()


# <a id="9"></a> 
# ***Confusion Matrix - Classification of Model Evaluation***<br/>
# 
# Confusion matrix shows us performance of a classification model. That means details of accuracy. For Example : 
# 
# ![confusionMatrix.PNG](http://i68.tinypic.com/23tqrlg.png)
# 
# **True Positive** : We predicted stars and we have 1248 stars<br/>
# **True Negative** : We predicted galaxies and we have 1489 galaxies<br/>
# **False Positive** : We predicted star, but it is not a star. It is a galaxy (Also known as a "Type I error.)<br/>
# **False Negative** : We predicted galaxy, but it is not a galaxy. It is a star (Also known as a "Type II error.)<br/>
# **Precision** = tp / (tp+fp))<br/>
# **Recall** = tp / (tp+fn))<br/>
# **f1** = 2 precision recall / ( precision + recall))<br/>

# **Confusion Matrix For Random Forest Classification**

# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = rf.predict(x_test)
y_actual = y_test
cm = confusion_matrix(y_actual, y_pred)


# In[ ]:


f, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
plt.xlabel("y_pred -> STAR = 0, GALAXY = 1")
plt.ylabel("y_actual -> STAR = 0, GALAXY = 1")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test, y_pred))


# <a id="10"></a> 
# # Conclusion

# According to [Comparison of Classifiers](#8) section DecisionTreeClassifier is the best option for this test data.  
# 
# *If you have a suggestion, I'd be happy to read it.*
