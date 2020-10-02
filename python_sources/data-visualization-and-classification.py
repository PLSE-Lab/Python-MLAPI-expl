#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import random
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import data
data = pd.read_csv('../input/data.csv')
data.head()


# The feature **Unnamed: 32** contains lots of NaN values, so I decide to drop them. In addition, feature **id** is not related to breast cancer, so it will be removed.

# In[ ]:


data.drop('Unnamed: 32', axis=1,inplace = True)
data.drop('id', axis=1,inplace = True)


# In[ ]:


#plot the PCC figure
corr = data.corr(method = 'pearson')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 275, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)


# Some features show strong correlation with each other. In order to reduce the dimensions, some features are dropped. For features **radius**, **perimeter** and **area**, I choose **area**. For **concavity, concave point** and **compatiness**, I choose concavity. For [**texture_worst, texture_mean**] and [**area_worst, area_mean**], I choose **texture_mean** and **texture_mean**. Therefore, 17 features are left.

# In[ ]:


#drop those related features
drop_features = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
              'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',
              'compactness_se','concave points_se','texture_worst','area_worst']
data1 = data.drop(drop_features, axis = 1)


# In[ ]:


#replace M and B in 'diagnosis' with 1 and 0 respectively for later classification problem
data1["class"] = data['diagnosis'].map({'M':1, 'B':0})
data1 = data1.drop('diagnosis', axis=1, inplace=True)
x = data1.copy(deep = True)
x = x.drop('class', axis=1, inplace=True)


# ## Feature Selection

# In[ ]:


x = x
y = data1['class'] 
#calculate the scores for each features in order to find out which features are more important.
feature_ranking = SelectKBest(chi2, k=5)
fit = feature_ranking.fit(x, y)

fmt = '%-8s%-20s%s'

print(fmt % ('', 'Scores', 'Features'))
for i, (score, feature) in enumerate(zip(feature_ranking.scores_, x.columns)):
    print(fmt % (i, score, feature))


# We can see that features: **area_mean, area_se, texture_mean, concavity_worst, concavity_mean** are the best 5 features.

# ## Min Max Normalization
# For machine learning methods, it is beneficial to do the normalization first. Before normaliztion, the ranges of different features are quite different. After min-max normalization, the interval is between [0, 1]. This makes the values invariant to rigid displacement of coordinates. However, it may encounter an out-of-bounds error if a future input case for normalization falls outside of the original data range.

# In[ ]:


data_norm = (data1 - data1.min()) / (data1.max() - data1.min())
data_norm.head()


# ## PCA

# In[ ]:


X_norm = data_norm
y_norm = data_norm['class']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
#PCA model, after normalization
pca = PCA(n_components=2)
X_r = pca.fit(X_norm).transform(X_norm)

#get the first class
data_0 = []
for i, label in enumerate(y_norm):
    if label == 0:
        data_0.append(X_r[i].tolist())
        
data_0_array = np.asarray(data_0)
 #get the second class
data_1 = []
for i, label in enumerate(y_norm):
    if label == 1:
        data_1.append(X_r[i].tolist())
        
data_1_array = np.asarray(data_1)
 #plot these two classes in one single plot
ax1.scatter(x=data_0_array[:,0], y=data_0_array[:,1], c='purple', label='Benign')
ax1.legend()
ax1.scatter(x=data_1_array[:,0], y=data_1_array[:,1], c='yellow', label='Malignant')
ax1.legend()
ax1.set_title('Principal Component Analysis after normalization (PCA)')
ax1.set_xlabel('1st principal component')
ax1.set_ylabel('2nd principal component')

#PCA model, before normalization

X = data1
y = data1['class']

pca = PCA(n_components=2)
X_r1 = pca.fit(X).transform(X)

data1_0 = []
for i, label in enumerate(y):
    if label == 0:
        data1_0.append(X_r1[i].tolist())
        
data1_0_array = np.asarray(data1_0)

data1_1 = []
for i, label in enumerate(y):
    if label == 1:
        data1_1.append(X_r1[i].tolist())
        
data1_1_array = np.asarray(data1_1)

ax2.scatter(x=data1_0_array[:,0], y=data1_0_array[:,1], c='purple', label='Benign')
ax2.legend()
ax2.scatter(x=data1_1_array[:,0], y=data1_1_array[:,1], c='yellow', label='Malignant')
ax2.legend()
ax2.set_title('Principal Component Analysis before normalization (PCA)')
ax2.set_xlabel('1st principal component')
ax2.set_ylabel('2nd principal component')


# From these two figures, it is very clear that normalization is quite essential for PCA. Before normalization, the distance between two classes is very close. Some of them are overlapped, so it is hard to classify them. However, after normalization, the distance is large enough to easily distinguish two classes.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.33, random_state=42)
print(X_train.shape)


# ## KNN after PCA

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))


# In[ ]:


confusion_matrix = confusion_matrix(y_test, pred)
print(confusion_matrix)


# The confusion matrix means that 121+67 test samples are correctly classified. 0 test sample is incorrectly classified, so the accuracy is 100%.

# In[ ]:


plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, label=pred)
plt.title('Classification using KNN', fontsize=12)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.legend(labels=pred)


# ## SVM after PCA

# In[ ]:


clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print((accuracy_score(y_test, pred)))

plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, label=pred)
plt.title('Classification using SVM', fontsize=12)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.legend(labels=pred)


# ## Naive Bayes Classifier

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.33, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred= gnb.predict(X_test)

score = accuracy_score(y_test, y_pred, normalize = True)
print(score)


# The above comparision is based on dimensional reduction. But what if I do the classification without reducing the dimensions?

# ## KNN without dimensional reduction

# In[ ]:


X = data_norm
y = data_norm['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))


# ## SVM without dimensional reduction

# In[ ]:


clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print((accuracy_score(y_test, pred)))


# In[ ]:


sns.countplot(data1['class'],label="Count")


# ## Naive Bayes Classifier without dimensional reduction

# In[ ]:


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
score = accuracy_score(y_test, y_pred, normalize = True)
print(score)


# ## Logistic regression without dimensional reduction

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
log_score = logreg.score(X_test, y_test)
print(log_score)


# KNN, SVM, Logistic Regression and Naive Bayes Classifier all get a very high accuracy. I guess one of the reasons is that the training samples are not large enough. Besides, the features that we chosen are very related to the classification. As long as the features are chosen correctly, the classification accuracy would be higher and dimension reduction doesn't matter a lot in this case.

# In[ ]:




