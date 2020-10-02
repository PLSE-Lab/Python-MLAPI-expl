#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# load the dataset
dataset = pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")


# In[ ]:


# see first 5 rows
dataset.head()


# In[ ]:


# dataset shape
dataset.shape


# In[ ]:


# check the null values
dataset.isna().any()
# So, their is no null value


# In[ ]:


# information about dataset
dataset.info()


# In[ ]:


# Check the unique value
dataset.target.unique()


# In[ ]:


# import some visualization library
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# visualize total count of sex
sns.countplot(x=dataset["sex"], data=dataset)


# In[ ]:


# But Its a ques that sex column contains two value, 0, 1
# Which one is male and which one is female?
# Men have a greater risk of heart attack than women do, and men have attacks earlier in life. Even after women reach the age of menopause, when women's death rate from heart disease increases, women's risk for heart attack is less than that for men.
# google.com # heart.org


# In[ ]:


sns.barplot(x=dataset["sex"], y=dataset["target"], data=dataset)


# In[ ]:


# According to google i think 0 is male and 1 is female


# In[ ]:


sns.set_style('whitegrid')
g = sns.FacetGrid(dataset, hue="target", palette="coolwarm", size=6, aspect=2)
g.map(plt.hist, 'age', bins=20, alpha=0.7)
plt.legend()


# In[ ]:


# lmplot
sns.set_style('whitegrid')
sns.lmplot('target', 'age', data=dataset, hue='sex', palette='coolwarm', size=6, aspect=1, fit_reg=False)


# In[ ]:


# visualize the correation

plt.figure(figsize=(12, 6))
g = dataset.corr()
data = g.index
sns.heatmap(dataset[data].corr(), annot=True)


# In[ ]:


# we see that cp(chest pain) is highly positive correlation with target
# thalach(maximum heart rate achieved) is also correlated with target
# exang(exercise induced angina) is highly negative correalted


# In[ ]:


corr_mat = dataset.corr()


# In[ ]:


corr_mat["target"].sort_values(ascending=False)


# In[ ]:


# lets split the dataset
X = dataset.drop("target", axis=1)
y = dataset["target"]
seed = 42
size = 0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=size)


# In[ ]:


# length of train and test
print(len(X_train))
print(len(X_test))


# In[ ]:


# import necessary packages

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVC': SVC(kernel="linear"),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'GNB': GaussianNB()
}

# fit and predict 
for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Random Forest":
        randomforest = clf.fit(X_train, y_train).predict(X_test)
    elif clf_name == "SVC":
        svc = clf.fit(X_train, y_train).predict(X_test)
    elif clf_name == "GNB":
        gnb = clf.fit(X_train, y_train).predict(X_test)
    elif clf_name == "KNN":
        knn = clf.fit(X_train, y_train).predict(X_test)


# In[ ]:


# predict SVC
svc


# In[ ]:


# predict knn
knn


# In[ ]:


# predict random forest
randomforest


# In[ ]:


# predict GNB
gnb


# In[ ]:


# check accuracy
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:



print("RandomForest accuracy: {}".format(accuracy_score(y_test, randomforest)))
print("SVC accuracy: {}".format(accuracy_score(y_test, svc)))
print("knn accuracy: {}".format(accuracy_score(y_test, knn)))
print("GNB accuracy: {}".format(accuracy_score(y_test, gnb)))


# In[ ]:


# Gaussian Naive Bayes perform very well, around 84%


# In[ ]:


print("RandomForest confusion_matrix: {}".format(confusion_matrix(y_test, randomforest)))
print("SVC confusion_matrix: {}".format(confusion_matrix(y_test, svc)))
print("knn confusion_matrix: {}".format(confusion_matrix(y_test, knn)))
print("GNB confusion_matrix: {}".format(confusion_matrix(y_test, gnb)))


# In[ ]:


# Store as joblib file


# In[ ]:


from joblib import load, dump


# In[ ]:


dump(clf, "model.joblib")


# In[ ]:


# load and predict by naivebayes classifier


# In[ ]:


from joblib import dump, load
c = load('model.joblib')
predict = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
clf.predict(predict)


# # I tried to predict in easy way and want to make it full clear.
# 
# # Please upvote me, It will encourage me more

# In[ ]:




