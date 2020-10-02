#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#To ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


# Reading File

# In[ ]:


mushroom = pd.read_csv('../input/mushrooms.csv')
mushroom.head(5)


# Importing libraries

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Checking for missing values

# In[ ]:


mushroom.isnull().sum()


# Check fo dimension of dataset and values of target class

# In[ ]:


mushroom['class'].unique()


# In[ ]:


mushroom.shape


# Models to apply - PCA, Logistic Regression, Random Forest, SVM, Naive Bayes Classifer and Decision Tree

# Encoding data from strings to intergers

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in mushroom.columns:
    mushroom[col] = labelencoder.fit_transform(mushroom[col])

mushroom.head()


# In[ ]:


mushroom['odor'].unique()


# In[ ]:


mushroom.groupby('class').size()


# Data visualizations

# In[ ]:


ax = sns.boxplot(x='class',y='stalk-color-above-ring',data=mushroom)
ax = sns.stripplot(x='class',y='stalk-color-above-ring',data=mushroom,jitter=True,edgecolor='gray')


# Separating target class from features

# In[ ]:


X = mushroom.iloc[:,1:]
Y = mushroom.iloc[:,0]
X.head()


# In[ ]:


Y.head()


# In[ ]:


X.describe().transpose()


# Correlation between different variables - only showing values > 0.4 or < -0.4

# In[ ]:


X.corr()[(X.corr() < -0.4) | (X.corr() > 0.4)]


# Normalizing the data

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X #returns an array


# Applying PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[ ]:


#Checking covariance
covarinace = pca.get_covariance()
covarinace


# In[ ]:


variance_explained = pca.explained_variance_
variance_explained


# In[ ]:


#Plotting graph to see which component has highest explained variance
with plt.style.context('dark_background'):
    plt.figure(figsize=(6,4))
    plt.bar(range(22),variance_explained,alpha=0.5,align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


#Plotting graph to see the cumulative sum of covariance and decide how many components to keep
with plt.style.context('dark_background'):
    plt.figure(figsize=(6,4))
    plt.bar(range(22),variance_explained.cumsum(),alpha=0.5,align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# Taking only first two components now - to easliy visualize the clustering

# In[ ]:


n = mushroom.values
pca = PCA(n_components=2)
x = pca.fit_transform(n)
plt.figure(figsize=(5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# Applying k-means clustering

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2,random_state=5)
X_clustered = kmeans.fit_predict(n)

LABEL_COLOR_MAP = {
    0: 'g',
    1: 'y'
}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize=(5,5))
plt.scatter(x[:,0],x[:,1],c=label_color)
plt.show()


# Data Partitioning into training and testing set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,
                                                    random_state=4)


# Default logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

LR_model = LogisticRegression()


# In[ ]:


LR_model.fit(X_train,y_train)


# In[ ]:


#Positive class prediction probabilities
y_prob = LR_model.predict_proba(X_test)[:,1]
#To derive class from the probabilities. If probability > 0.5, assign class 1 otherwise class 0
y_pred = np.where(y_prob > 0.5, 1, 0)
LR_model.score(X_test, y_pred)


# In[ ]:


#Making confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


#Geting ROC metric - Area under the curve value
auc_roc = metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],linestyle = '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# 

# Logistic Regression (tuned)

# Tuned parameters include applying different learning rate (C) and different penalty types. L1(Lasso) is the first moment norm |x1-x2| (|w| for regularization case). L2(Ridge) is the second moment norm |x1-x2|^2 (|w|^2 for regularization). L2 shrinks all the coefficient by the same proportions but eliminates none, while L1 can shrink some coefficients to zero, performing variable selection. If all the features are correlateed with the target, L2 performs L1. If only a subset of the features are correlated to the target, L1 outperforms L2.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

LR_model = LogisticRegression()

tuned_parameters = {'C': [0.001,0.01,0.1,1,10,100,1000], 
                    'penalty':['l1','l2']
                   }


# In[ ]:


mushroom.corr()


# The grid search provided by GridSearchCV generates all possible candidates from a grid of parameter values specified with the tuned_parameter. The GridSearchCV instance implements the usual estimator API: when 'fitting' it on a dataset, all the possible combinations of parameter vaues are evaluated and the best combination is retained.

# In[ ]:


from sklearn.model_selection import GridSearchCV
LR = GridSearchCV(LR_model, tuned_parameters, cv=10)


# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


print(LR.best_params_)


# In[ ]:


#Probability for being in positive class
y_prob = LR.predict_proba(X_test)[:,1]
#Getting class - if probability > 0.5, true(1) class otherwise false(0) class
y_pred = np.where(y_prob > 0.5,1,0)
LR.score(X_test,y_pred)


# In[ ]:


#Confusion matrix for the model
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


#Precision, Recall & f1 scores
measures = metrics.classification_report(y_test,y_pred)
measures


# In[ ]:


#ROC value - Area under the curve (AUC) value
auc_roc = metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


LR_ridge = LogisticRegression(penalty='l2')
LR_ridge.fit(X_train,y_train)


# In[ ]:


#Probability for being in positive class
y_prob = LR_ridge.predict_proba(X_test)[:,1]
#Getting class - if probability > 0.5, true(1) class otherwise false(0) class
y_pred = np.where(y_prob > 0.5,1,0)
LR_ridge.score(X_test,y_pred)


# In[ ]:


#Confusion matrix for the model
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


#ROC value - Area under the curve (AUC) value
auc_roc = metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
naive_model = GaussianNB()
naive_model.fit(X_train,y_train)


# In[ ]:


#Probability for being in positive class
y_prob = naive_model.predict_proba(X_test)[:,1]
#Getting class - if probability > 0.5, true(1) class otherwise false(0) class
y_pred = np.where(y_prob > 0.5,1,0)
naive_model.score(X_test,y_pred)


# In[ ]:


print("Number of mislabel points from %d points: %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


# In[ ]:


#Cross-validating 10 times and calculating their accuracy
scores = cross_val_score(naive_model, X, Y, cv=10, scoring='accuracy')
print(scores)


# In[ ]:


scores.mean()


# In[ ]:


#Confusion matrix for the model
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


#Precision, Recall & f1 scores
measures = metrics.classification_report(y_test,y_pred)
measures


# In[ ]:


#ROC value - Area under the curve (AUC) value
auc_roc = metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
svm_model = SVC()


# Tunning SVM on 2 parameters gamma & C
# gamma parameter is the inverse of the radius of influence of samples selected by the model as support vectors. In other words, it defines how far the influence of a single training example reaches. Low gamma values means influence is far and wide whereas high gamma values means influences is near and close.
# C parameter defines tradeoff between misclassification rate and complexity of the decision surface. Low value of C means simple(smooth) decision surface and high misclassification rate whereas high value of C means low misclassification rate and complex decision surface.

# SVM with non-polynomial kernel

# In[ ]:


tuned_parameters = {
    'C' : [1, 10, 100, 500, 1000],'kernel': ['linear','rbf'],
    'C' : [1, 10, 100, 500, 1000],'gamma': [1, 0.1, 0.01, 0.0001], 'kernel':['rbf']
}
tuned_parameters


# RandomizedSearchCV implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. This has two main benefits over an exhaustive search: 1)A budget can be chosen independent of the number of parameters and possible values. 2)Adding parameters that do not influence the performance does not decrease efficiency.
# 
# 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters, cv = 10, 
                               scoring='accuracy', n_iter = 20)


# Might take some time

# In[ ]:


model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[ ]:


print(model_svm.best_params_)


# In[ ]:


y_pred = model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# SVM with polynomial kernel

# In[ ]:


tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
}
tuned_parameters


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters, cv = 10, 
                               scoring='accuracy', n_iter = 20)


# In[ ]:


#might take some time
model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[ ]:


print(model_svm.best_params_)


# In[ ]:


#print(model_svm.grid_scores_)


# In[ ]:


y_pred = model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)


# In[ ]:


#Probabilities for positive class
y_prob = RF_model.predict_proba(X_test)[:,1]
#Class from probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)
RF_model.score(X_test,y_test)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# (Just for practice) Tuning Random Forest on 3 features 
# 1. max_features - maximum number of features Random Forest is allowed to try in individual tree - i) auto - take all features ii)sqrt - take square root of the number of features iii)log2 - take log base 2 of the number of features.
# *Higher the number of features, slower is the algorithm, better is the performance(output model)
# 2. n_estimators - number of trees to build 
# 3. min_samples_leaf - purity of leaf node

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RFC_model = RandomForestClassifier()
tuned_parameters = { 'min_samples_leaf': range(10,100,10),
                   'n_estimators': range(10,100,10),
                   'max_features': ['auto','sqrt','log2']
                   }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

RFC_model = RandomizedSearchCV(RFC_model, tuned_parameters,cv=10, 
                              scoring='accuracy',
                              n_iter=20, n_jobs = -1)
#n_jobs tells the engine how many processors is it allowed to use. If -1, no restrictions, if 1, can only use 1 processor


# In[ ]:


RFC_model.fit(X_train, y_train)


# In[ ]:


print(RFC_model.best_score_)


# In[ ]:


print(RFC_model.best_params_)


# In[ ]:


#probabilities for being in positive class
y_prob = RFC_model.predict_proba(X_test)[:,1]
#getting class from probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)
RFC_model.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Decision Tree model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()


# In[ ]:


tree_model.fit(X_train, y_train)


# In[ ]:


#probabilities for being in positive class
y_prob = tree_model.predict_proba(X_test)[:,1]
#getting class from probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)
tree_model.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# (Just for practice) Tuning 3 parameters of Decision Tree 
# 1. Criterion - selection of split - gini index or entropy
# 2. max-depth - maximum vertical depth of tree - to control overfitting of tree
# max_features and min_samples_leaf - same as Random Forest

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier()

tuned_parameters = { 'criterion': ['gini','entropy'], 
                    'max_features': ['auto','sqrt','log2'],
                   'min_samples_leaf': range(1,100,1),
                   'max_depth': range(1,50,1)
                   }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

DT_model = RandomizedSearchCV(DT_model, tuned_parameters, cv=10, scoring='accuracy',
                             n_iter=20,n_jobs=-1,random_state=5)


# In[ ]:


DT_model.fit(X_train,y_train)


# In[ ]:


print(DT_model.best_score_)


# In[ ]:


print(DT_model.best_params_)


# In[ ]:


#probabilities for being in positive class
y_prob = DT_model.predict_proba(X_test)[:,1]
#getting class from probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)
DT_model.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


#Getting values for the ROC curve and finding the best AUC value
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc


# In[ ]:


#Plotting the roc curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate,color='red',
        label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle= '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Neural Network

# In[ ]:




