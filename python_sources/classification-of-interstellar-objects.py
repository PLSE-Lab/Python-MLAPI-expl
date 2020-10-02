#!/usr/bin/env python
# coding: utf-8

# [](http://)Classification of objects as galaxies, stars or quasars.

# ## Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data into a Pandas dataframe

# In[ ]:


df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv', delimiter=',', header=1)
df.head()


# Details on the attributes and visualizations are clearly documented in this [kernel](https://www.kaggle.com/lucidlenn/data-analysis-and-classification-using-xgboost) by [LennartGrosser](https://www.kaggle.com/lucidlenn) and hence, no extensive exploration will be done.

# * Drop the unnecessary columns from the dataframe.

# In[ ]:


df.drop(['objid', 'specobjid', 'run', 'rerun', 'camcol', 'plate', 'mjd', 'fiberid', 'field'], axis=1, inplace=True)


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


f, ax = plt.subplots(1, figsize=(8,8))
sns.heatmap(df.corr(), annot=True, ax=ax)


# * *g, r, i, z* are highly correlated and PCA could be used for dimensionality reduction.
# * *u* also shows similar behavior but the correlation drops as one moves from *g* to *z*.

# ### PCA
# * The four highly correlated columns are reduced to only a single column.

# In[ ]:


pca_val = df[['g', 'r', 'i', 'z']]
pca = PCA(n_components=1)
pca_val = pca.fit_transform(pca_val)
pca_val = pca_val.reshape(-1,)


# In[ ]:


pca_val = pd.Series(pca_val, name='pca_val')
df = pd.concat([df, pca_val], axis=1)
df.drop(['g', 'r', 'i', 'z'], axis=1, inplace=True)
df.head()


# ### Prepare data for machine learning

# In[ ]:


X = df.drop('class', axis=1)
y = df['class']
y.value_counts()


# * Split the data into training set, dev set and the test set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.15, random_state=5)
X_train_feature_imp = X_train
X_train = X_train.values
X_dev = X_dev.values
X_test = X_test.values


# In[ ]:


print(y_train.value_counts())
print(y_dev.value_counts())
print(y_test.value_counts())


# * All three splits contain an even distribution of data.
# * The three labels could be encoded into integers using Scikit-Learn's label encoder.

# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)


# ### Machine Learning
# * Four classifiers and one voting classifier was chosen to perform the classification task.
# * Validation curves were plotted between the performance metric and the hyperparameters to understand the optimal values for hyperparameters. 

# #### Logistic regression

# In[ ]:


C = np.logspace(-1, 4, 10)
accuracy_train = []
accuracy_dev = []
for i in C:
    clf = LogisticRegression(C=i)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.semilogx(C, accuracy_train, label='Train score')
plt.semilogx(C, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * Performance peaked when the vaue of C was around 1000.

# In[ ]:


clf = LogisticRegression(C=1000)
clf.fit(X_train, y_train)
print('Training set: ')
print('Accuracy: {}'.format(accuracy_score(y_train, clf.predict(X_train))))
print('Classification report:\n{}'.format(classification_report(y_train, clf.predict(X_train))))
print('\nDev set: ')
print('Accuracy: {}'.format(accuracy_score(y_dev, clf.predict(X_dev))))
print('Classification report:\n{}'.format(classification_report(y_dev, clf.predict(X_dev))))


# #### Support Vector Machine

# In[ ]:


C = np.logspace(-1, 4, 5)
accuracy_train = []
accuracy_dev = []
for i in C:
    clf = SVC(C=i)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.semilogx(C, accuracy_train, label='Train score')
plt.semilogx(C, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * The performance on the dev set peaks between 100 and 1000.
# * 500 was chosen for the value of C.
# * *Further plots could be drawn between 100 and 1000 to figure out the exact value.*

# In[ ]:


gamma = np.logspace(-2, 1, 5)
accuracy_train = []
accuracy_dev = []
for i in gamma:
    clf = SVC(C = 500,gamma=i)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.semilogx(gamma, accuracy_train, label='Train score')
plt.semilogx(gamma, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * gamma value was taken as 0.001.

# In[ ]:


clf = SVC(C=500, gamma=0.001)
clf.fit(X_train, y_train)
print('Training set: ')
print('Accuracy: {}'.format(accuracy_score(y_train, clf.predict(X_train))))
print('Classification report:\n{}'.format(classification_report(y_train, clf.predict(X_train))))
print('\nDev set: ')
print('Accuracy: {}'.format(accuracy_score(y_dev, clf.predict(X_dev))))
print('Classification report:\n{}'.format(classification_report(y_dev, clf.predict(X_dev))))


# #### Random Forest Classifier

# In[ ]:


n_estimators = [100, 200, 300, 400, 500]
accuracy_train = []
accuracy_dev = []
for i in n_estimators:
    clf = RandomForestClassifier(n_estimators=i, random_state=10)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.plot(n_estimators, accuracy_train, label='Train score')
plt.plot(n_estimators, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * Values greater than 300 estimators showed better performance.
# * 300 was chosen as the final value (faster computation).
# * Other parameters could also be tweaked for better performance on the dev set.

# In[ ]:


clf = RandomForestClassifier(n_estimators=300, random_state=10)
clf.fit(X_train, y_train)
print('Training set: ')
print('Accuracy: {}'.format(accuracy_score(y_train, clf.predict(X_train))))
print('Classification report:\n{}'.format(classification_report(y_train, clf.predict(X_train))))
print('\nDev set: ')
print('Accuracy: {}'.format(accuracy_score(y_dev, clf.predict(X_dev))))
print('Classification report:\n{}'.format(classification_report(y_dev, clf.predict(X_dev))))


# #### Decision Tree Classifier

# In[ ]:


min_samples_split = [2, 3, 5, 10]
accuracy_train = []
accuracy_dev = []
for i in min_samples_split:
    clf = DecisionTreeClassifier(min_samples_split=i, random_state=25, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.plot(min_samples_split, accuracy_train, label='Train score')
plt.plot(min_samples_split, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('Minimum samples split')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * 2 was chosen as the hyperparameter by looking at the performance on the training set as the dev set performance was similar.

# In[ ]:


min_samples_leaf = [1, 2, 3, 5, 10]
accuracy_train = []
accuracy_dev = []
for i in min_samples_leaf:
    clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=i, random_state=25)
    clf.fit(X_train, y_train)
    accr = accuracy_score(y_train, clf.predict(X_train))
    accuracy_train.append(accr)
    accr = accuracy_score(y_dev, clf.predict(X_dev))
    accuracy_dev.append(accr)
f, ax = plt.subplots(1, figsize=(8,6))
plt.plot(min_samples_leaf, accuracy_train, label='Train score')
plt.plot(min_samples_leaf, accuracy_dev, label='Dev score')
plt.legend(loc='best')
plt.xlabel('Minimum samples leaf')
plt.ylabel('Accuracy')
plt.title('Performance on Training set and Dev set')


# * 9 was chosen for the min_samples_leaf hyperparameter as it showed better performance on the validation set.

# In[ ]:


clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=9, random_state=25)
clf.fit(X_train, y_train)
print('Training set: ')
print('Accuracy: {}'.format(accuracy_score(y_train, clf.predict(X_train))))
print('Classification report:\n{}'.format(classification_report(y_train, clf.predict(X_train))))
print('\nDev set: ')
print('Accuracy: {}'.format(accuracy_score(y_dev, clf.predict(X_dev))))
print('Classification report:\n{}'.format(classification_report(y_dev, clf.predict(X_dev))))


# * Random Forest and Decision Tree showed better performance over SVM and logistic regression.

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=300, random_state=10)
dt_clf = DecisionTreeClassifier(min_impurity_split=2, min_samples_leaf=9, random_state=25)
lr_clf = LogisticRegression(C=1000)
e_vclf = VotingClassifier([('lr', lr_clf), ('rf', rf_clf), ('dt', dt_clf)], voting='soft')


# In[ ]:


e_vclf.fit(X_train, y_train)
print('Training set: ')
print('Accuracy: {}'.format(accuracy_score(y_train, e_vclf.predict(X_train))))
print('Classification report:\n{}'.format(classification_report(y_train, e_vclf.predict(X_train))))
print('\nDev set: ')
print('Accuracy: {}'.format(accuracy_score(y_dev, e_vclf.predict(X_dev))))
print('Classification report:\n{}'.format(classification_report(y_dev, e_vclf.predict(X_dev))))


# * The classifier created using *soft voting* showed poorer performance on the dev set than Random Forest and Decision Tree classifiers.
# * Looking at the classification reports from all the classifiers, Decision Tree classifier was chosen for the final model as it had shown good numbers for precision, recall and the f1 score.

# In[ ]:


clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=9, random_state=25)
clf.fit(X_train, y_train)
print('Test set: ')
print('Accuracy: {}'.format(accuracy_score(y_test, clf.predict(X_test))))
print('Classification report:\n{}'.format(classification_report(y_test, clf.predict(X_test))))


# In[ ]:


feature_imp = pd.DataFrame({'Feature': X_train_feature_imp.columns,'Importance': clf.feature_importances_})
f = plt.subplots(1, figsize=(8,6))
plt.bar(feature_imp.Feature, feature_imp.Importance, log=True, alpha=0.8, width=0.5, edgecolor='k')
plt.ylim(1e-5, 2)
plt.title('Feature Importance')

