#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.count()


# In[ ]:


df_train[df_train['Sex'].str.match("female")].count()


# In[ ]:


df_train[df_train['Sex'].str.match("male")].count()


# In[ ]:


df_train['Pclass'].value_counts()


# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data=df_train)


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data=df_train)


# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass',y='Age',data=df_train)


# In[ ]:


def check_age(age):
    if pd.isnull(age):
        return int(df_train["Age"].mean())
    else:
        return age


# In[ ]:


df_train['Age'] = df_train['Age'].apply(check_age)


# In[ ]:


df_train['Age'].isnull().sum()


# In[ ]:


df_train.drop(["Cabin","Name"],inplace=True,axis=1)


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


pd.get_dummies(df_train["Sex"]).head()


# In[ ]:


sex = pd.get_dummies(df_train["Sex"])


# In[ ]:


embarked = pd.get_dummies(df_train["Embarked"])


# In[ ]:


pclass = pd.get_dummies(df_train["Pclass"])


# In[ ]:


df_train = pd.concat([df_train,pclass,sex,embarked],axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train.drop(["PassengerId","Pclass","Sex","Ticket","Embarked"],axis=1,inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


X = df_train.drop("Survived",axis=1)
y = df_train["Survived"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[ ]:


#from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X_train,y_train)
y_pred_lr = lrmodel.predict(X_test)

accuracy_score_lr = accuracy_score(y_pred_lr,y_test)
accuracy_score_lr


# In[ ]:


#predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[ ]:


# Decision Tree
dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state = 0)
dtree.fit(X_train,y_train)
y_pred_dtree = dtree.predict(X_test)


# In[ ]:


accuracy_score_dtree = accuracy_score(y_pred_dtree,y_test)
accuracy_score_dtree


# In[ ]:


# Random Forest
rf = RandomForestClassifier(criterion = 'gini',random_state = 0)
rf.fit(X_train,y_train)


# In[ ]:


y_pred_rf = rf.predict(X_test)
accuracy_score_rf = accuracy_score(y_pred_rf,y_test)
accuracy_score_rf


# In[ ]:


sv = svm.SVC(kernel= 'linear',gamma =2)
sv.fit(X_train,y_train)


# In[ ]:


#SVM
y_pred_svm = sv.predict(X_test)
accuracy_score_svm = accuracy_score(y_pred_svm,y_test)
accuracy_score_svm


# In[ ]:


#KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)


# In[ ]:


y_pred_knn = knn.predict(X_test)
accuracy_score_knn = accuracy_score(y_pred_knn,y_test)
accuracy_score_knn


# In[ ]:


scores = [accuracy_score_lr,accuracy_score_dtree,accuracy_score_rf,accuracy_score_svm,accuracy_score_knn]
scores = [i*100 for i in scores]
algorithm  = ['Logistic Regression','Decision Tree','Random Forest','SVM', 'K-Means']
index = np.arange(len(algorithm))
plt.bar(index, scores)
plt.xlabel('Algorithm', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=5)
plt.xticks(index, algorithm, fontsize=10, rotation=30)
plt.title('Accuracy scores for each classification algorithm')
plt.ylim(80,100)
plt.show() 


# In[ ]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()

