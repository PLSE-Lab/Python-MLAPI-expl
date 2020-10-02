#!/usr/bin/env python
# coding: utf-8

# <h1><div style="text-align: center"> Human Activity Recognition using Machine Learning </div></h1> 

# ## Contents
# <h3>1. Data Description </h3> 
#  --- (1) Feature Description<br />
#  --- (2) Class Description<br />
#  
# &ast; Baseline performance measure
# <h3>2. Preprocessing </h3> 
# 
# <h3>3. Feature Selection </h3> 
#  --- (1) RFECV (wrapper method)<br />
#  --- (3) Embedded method<br />
#  <h3>4. Model selection</h3> 
#  --- (1) Basic classifiers (Decision Tree, KNN, MLP, SVM)<br />
#  --- (2) Ensemble (RandomForest, Bagging, Boosting, Stacking)<br />
# <br /><br />

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier

from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Data Description
# 
# The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years.<br />
# Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist.<br />
# https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones

# In[ ]:


data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


data.head()


# In[ ]:


X = data.iloc[:,:-2] # features
y = data.iloc[:,-1]  # class

X_test = test_data.iloc[:,:-2] # features
y_test = test_data.iloc[:,-1]  # class


# ### 1 - (1) Feature Description

# In[ ]:


X.info()


# In[ ]:


feature_names = X.columns


# Total number of features : 561, including<br />
# triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration,<br />
# triaxial Angular velocity from the gyroscope,<br />
# a 561-feature vector with time and frequency domain variables.

# ### 1 - (2) Class Description
# Activity class<br />
# 6 classes : WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING <br />

# In[ ]:


data.Activity.unique()


# In[ ]:


class_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']


# In[ ]:


y.value_counts()


# In[ ]:


plt.figure(figsize=(14,5))
ax = sns.countplot(y, label = "Count", palette = "Set3")
LAYING, STANDING, SITTING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS = y.value_counts()


# ### &ast; Baseline performance measure

# 
# Dummy classifier is used to measure baseline performance.<br />
# DummyClassifier is a classifier that makes predictions using simple rules.<br />
# Below are the baseline performance using "most_frequent" strategy.<br />
# The model that will be developed later on should show better performance than this.<br />

# In[ ]:


dummy_classifier = DummyClassifier(strategy="most_frequent")
y_pred = dummy_classifier.fit(X, y).predict(X_test)
dummy_classifier.score(X_test, y_test)


# In[ ]:


cm = confusion_matrix(y_test, dummy_classifier.predict(X_test), labels = class_names)
sns.heatmap(cm, annot=True, fmt="d", xticklabels = class_names, yticklabels = class_names)


# The most frequent class in this training set is LAYING. <br />
# When you classify every instance in the training set into LAYING class, the accuracy is about 18.2%.

# ## 2. Preprocessing

# We don't need one-hot encoding here since all the features we've got are numerical values.<br />
# Also, no scaling is needed since all the values are already normalized within the range between -1 and 1.

# ## 3. Feature Selection

# Sometimes, it's better to use only some of the given features.<br />
# Too many features cause high complexity and overfitting!<br />
# 
# There are several feature selection methods,<br />
# (1) Feature subset selection,<br />
# (2) PCA,<br />
# 
# and among feature subset selection methods are filter method, wrapper method, embedded method.

# ### 3 - (1) RFECV (wrapper method)
# Recursive Feature Elimination(RFE) is a wrapper method for feature selection.<br />
# RFECV gives feature ranking with recursive feature elimination and the optimal number of features.

# In[ ]:


rf_for_refcv = RandomForestClassifier() 
rfecv0 = RFECV(estimator = rf_for_refcv, step = 1, cv = 5, scoring = 'accuracy')   #5-fold cross-validation
rfecv0 = rfecv0.fit(X, y)

print('Optimal number of features :', rfecv0.n_features_)
print('Best features :', X.columns[rfecv0.support_])


# In[ ]:


rfecv_X = X[X.columns[rfecv0.support_]]  # rfecv_X.shape = (7352, 550)
rfecv_X_test = X_test[X.columns[rfecv0.support_]]  # rfecv_X_test.shape = (2947, 550)


# ### 3 - (2) Embedded method

# Embedded method finds the optimal feature subset during model training.<br />
# Below is the Random Forest example. Random Forest is an ensemble learning method using multiple decision trees.<br />
# Constructed decision tree identifies the most significant variables and it gives us some information about feature importances.

# In[ ]:


rf_for_emb = RandomForestClassifier()      
rf_for_emb = rf_for_emb.fit(X, y)
importances = rf_for_emb.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_for_emb.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# In[ ]:


indices


# In[ ]:


h_indices = indices[:100]  # 100 best features
th_indices = indices[:300]  # 300 best features


# In[ ]:


plt.figure(1, figsize=(25, 13))
plt.title("Feature importances")
plt.bar(range(100), importances[h_indices], color="g", yerr=std[h_indices], align="center")
plt.xticks(range(100), X.columns[h_indices],rotation=90)
plt.xlim([-1, 100])
plt.show()


# In[ ]:


hundred_X = X[X.columns[h_indices]]
hundred_X_test = X_test[X.columns[h_indices]]

threeh_X = X[X.columns[th_indices]]
threeh_X_test = X_test[X.columns[th_indices]]


# ## 4. Model Selection

# ### 4 - (1) Basic Classifiers (Decision Tree, KNN, MLP, SVM)

# In[ ]:


decision_tree00 = tree.DecisionTreeClassifier()
dtclf00 = decision_tree00.fit(X, y)
dtclf00.score(X_test, y_test)


# In[ ]:


decision_tree01 = tree.DecisionTreeClassifier()
dtclf01 = decision_tree01.fit(hundred_X, y)
dtclf01.score(hundred_X_test, y_test)


# In[ ]:


decision_tree02 = tree.DecisionTreeClassifier()
dtclf02 = decision_tree02.fit(threeh_X, y)
dtclf02.score(threeh_X_test, y_test)


# In[ ]:


decision_tree03 = tree.DecisionTreeClassifier()
dtclf03 = decision_tree03.fit(rfecv_X, y)
dtclf03.score(rfecv_X_test, y_test)


# For some reason it doesn't seem like selected features guarantee better performance.

# In[ ]:


decision_tree04 = tree.DecisionTreeClassifier(min_samples_leaf=4)
dtclf04 = decision_tree04.fit(X, y)
dtclf04.score(X_test, y_test)


# In[ ]:


decision_tree05 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf05 = decision_tree05.fit(X, y)
dtclf05.score(X_test, y_test)


# In[ ]:


decision_tree06 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf06 = decision_tree06.fit(rfecv_X, y)
dtclf06.score(rfecv_X_test, y_test)


# In[ ]:


decision_tree07 = tree.DecisionTreeClassifier(min_samples_leaf=6)
dtclf07 = decision_tree07.fit(threeh_X, y)
dtclf07.score(threeh_X_test, y_test)


# 
# This code shows how the testing accuracy varies when K changes.

# In[ ]:


k_range = range(3, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knnclf = knn.fit(X, y)
    scores.append(knnclf.score(X_test, y_test))

plt.figure(figsize=(14,5))
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[ ]:


knn0 = KNeighborsClassifier(n_neighbors = 18)
knnclf0 = knn0.fit(X, y)
knnclf0.score(X_test, y_test)


# In[ ]:


knn1 = KNeighborsClassifier(n_neighbors = 19)
knnclf1 = knn1.fit(threeh_X, y)
knnclf1.score(threeh_X_test, y_test)


# In[ ]:


knn2 = KNeighborsClassifier(n_neighbors = 28)
knnclf2 = knn2.fit(rfecv_X, y)
knnclf2.score(rfecv_X_test, y_test)


# 
# 

# In[ ]:


mlp0 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp0 = mlp0.fit(X, y)
mlp0.score(X_test, y_test)


# In[ ]:


mlp1 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp1 = mlp1.fit(rfecv_X, y)
mlp1.score(rfecv_X_test, y_test)


# In[ ]:


mlp2 = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp2 = mlp2.fit(rfecv_X, y)
mlp2.score(rfecv_X_test, y_test)


# In[ ]:


mlp3 = MLPClassifier(hidden_layer_sizes=(15, 15, 15))
mlp3 = mlp3.fit(X, y)
mlp3.score(X_test, y_test)


# In[ ]:


mlp4 = MLPClassifier(hidden_layer_sizes=(20, 20))
mlp4 = mlp4.fit(X, y)
mlp4.score(X_test, y_test)


# In[ ]:


mlp5 = MLPClassifier(hidden_layer_sizes=(30, 30))
mlp5 = mlp5.fit(X, y)
mlp5.score(X_test, y_test)


# 
# 

# In[ ]:


svcclf0 = SVC()
svcclf0 = svcclf0.fit(X, y)
svcclf0.score(X_test, y_test)


# In[ ]:


svcclf1 = SVC()
svcclf1 = svcclf1.fit(hundred_X, y)
svcclf1.score(hundred_X_test, y_test)


# In[ ]:


svcclf2 = SVC()
svcclf2 = svcclf2.fit(threeh_X, y)
svcclf2.score(threeh_X_test, y_test)


# In[ ]:


svcclf3 = SVC()
svcclf3 = svcclf3.fit(rfecv_X, y)
svcclf3.score(rfecv_X_test, y_test)


# 
# 

# ### 4 - (2) Ensemble (RandomForest, Bagging, Boosting, Stacking)

# An obvious approach to making decisions more reliable is to combine the output of different models.<br />
# 
# RandomForest builds a randomized decision tree. ex) picking one of the N best options at random instead of a single winner.<br />

# In[ ]:


RFclf0 = RandomForestClassifier()
RFclf0 = RFclf0.fit(X, y)
RFclf0.score(X_test, y_test)


# In[ ]:


RFclf1 = RandomForestClassifier()
RFclf1 = RFclf1.fit(rfecv_X, y)
RFclf1.score(rfecv_X_test, y_test)


# In[ ]:


x_range = range(10, 25)
scores2 = []
for x in x_range:
    RFclf2 = RandomForestClassifier(n_estimators = x)
    RFclf2 = RFclf2.fit(X, y)
    scores2.append(RFclf2.score(X_test, y_test))

plt.plot(x_range, scores2)
plt.xlabel('Number of trees in the forest')
plt.ylabel('Testing Accuracy')


# In[ ]:


RFclf3 = RandomForestClassifier(n_estimators = 20)
RFclf3 = RFclf3.fit(X, y)
RFclf3.score(X_test, y_test)


# In[ ]:


RFclf4 = RandomForestClassifier(n_estimators = 20)
RFclf4 = RFclf4.fit(rfecv_X, y)
RFclf4.score(rfecv_X_test, y_test)


# In[ ]:


RFclf5 = RandomForestClassifier(n_estimators = 16)
RFclf5 = RFclf5.fit(threeh_X, y)
RFclf5.score(threeh_X_test, y_test)


# 
# BaggingClassifier makes the models vote. The models receive the equal weight.<br />
# It is suitable for unstable learning schemes like decision trees.

# In[ ]:


bag0 = BaggingClassifier()
bag0 = bag0.fit(X, y)
bag0.score(X_test, y_test)


# In[ ]:


bag1 = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3))
bag1 = bag1.fit(X, y)
bag1.score(X_test, y_test)


# In[ ]:


bag2 = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=28))
bag2 = bag2.fit(X, y)
bag2.score(X_test, y_test)


# In Boosting, each new model is influenced by the performance of those built previously.

# In[ ]:


ABclf0 = AdaBoostClassifier() #n_estimators=50, learning_rate = 1
ABclf0 = ABclf0.fit(X, y)
ABclf0.score(X_test, y_test)


# In[ ]:


ABclf1 = AdaBoostClassifier(n_estimators=200, learning_rate = 0.5)
ABclf1 = ABclf1.fit(X, y)
ABclf1.score(X_test, y_test)


# In[ ]:


ABclf2 = AdaBoostClassifier(n_estimators=400, learning_rate = 0.5)
ABclf2 = ABclf2.fit(X, y)
ABclf2.score(X_test, y_test)


#  
#  
# Finally I made the good models I found above vote.

# In[ ]:


voteclf1 = VotingClassifier(
    estimators=[('knn0', KNeighborsClassifier(n_neighbors = 19)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20))),
                ('RFclf1', RandomForestClassifier())], 
    voting='soft')
voteclf1 = voteclf1.fit(X, y)
voteclf1.score(X_test, y_test)


# This DeprecationWarning is some kind of bug and it's said it will have been fixed by August 2018.

# In[ ]:


voteclf2 = VotingClassifier(
    estimators=[('knn0', KNeighborsClassifier(n_neighbors = 18)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20))),
                ('RFclf1', RandomForestClassifier())], 
    voting='soft')
voteclf2 = voteclf2.fit(X, y)
voteclf2.score(X_test, y_test)


# In[ ]:


voteclf3 = VotingClassifier(
    estimators=[('svcclf0', SVC()),
                ('knn0', KNeighborsClassifier(n_neighbors = 18)),
                ('RFclf03', RandomForestClassifier(n_estimators = 20)), 
                ('mlp0', MLPClassifier(hidden_layer_sizes=(15, 15))),
                ('mlp4', MLPClassifier(hidden_layer_sizes=(20, 20)))],
                 
    voting='hard')
voteclf3 = voteclf3.fit(X, y)
voteclf3.score(X_test, y_test)


# ### Best Model
# 
# Voteclf2 showed the best performance above all!

# In[ ]:


cm = confusion_matrix(y_test, voteclf2.predict(X_test), labels = class_names)
sns.heatmap(cm, annot = True, fmt = "d", xticklabels = class_names, yticklabels = class_names)

