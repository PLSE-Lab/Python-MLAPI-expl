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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


import plotly.graph_objs as go


# In[ ]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


train_data = pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_train_data.csv')
eval1 = pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_eval_data.csv')
target = pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_target_data.csv')


# In[ ]:


train_data.describe()


# Correlation Matrix

# In[ ]:


#Correlation matrix
corrmat = train_data.corr()
fig = plt.figure(figsize = (16, 16))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[ ]:


cols=list(train_data.columns)
cols


# In[ ]:


colors=['#b84949', '#ff6f00', '#ffbb00', '#9dff00', '#329906', '#439c55', '#67c79e', '#00a1db', '#002254', '#5313c2', '#c40fdb', '#e354aa']


# Univariate Analysis

# In[ ]:


k=0
for i in cols:
    train_data.groupby(i)['CARAVAN'].count().plot(kind='bar', grid=True, color=colors[k%12],
    figsize=(10, 7)).set_ylabel('Count')
    #s='images/'+i+'.png'
    #plt.savefig(s)
    k+=1
    plt.show()


# In[ ]:


cols1=list(eval1.columns)


# To find how other parameters affect CARAVAN

# In[ ]:


for i in cols1:
    df=train_data.groupby(['CARAVAN', i])['CARAVAN'].count()
    df = df.unstack().fillna(0)
    ax = (df).plot(
    kind='bar',
    figsize=(10, 7),
    grid=True
    )
    ax.set_ylabel('Count')
    plt.show()


# In[ ]:


df_1=train_data.copy()


# In[ ]:


X = df_1.drop(['CARAVAN'], axis=1).values
y = df_1['CARAVAN'].values


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic Regression

# In[ ]:


lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
predictions = lr.predict(eval1)
print(accuracy_score(target, predictions))
print(confusion_matrix(target, predictions))
print(classification_report(target, predictions))


# Decision Tree

# In[ ]:


CART = DecisionTreeClassifier()
CART.fit(X_train, y_train)
predictions = CART.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


CART = DecisionTreeClassifier()
CART.fit(X_train, y_train)
predictions = CART.predict(eval1)
print(accuracy_score(target, predictions))
print(confusion_matrix(target, predictions))
print(classification_report(target, predictions))


# KNN

# In[ ]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
predictions = KNN.predict(eval1)
print(accuracy_score(target, predictions))
print(confusion_matrix(target, predictions))
print(classification_report(target, predictions))


# LDA

# In[ ]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(eval1)
print(accuracy_score(target, predictions))
print(confusion_matrix(target, predictions))
print(classification_report(target, predictions))


# SVM

# In[ ]:


SVM = SVC(gamma='auto')
SVM.fit(X_train, y_train)
predictions = SVM.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


SVM = SVC(gamma='auto')
SVM.fit(X_train, y_train)
predictions = SVM.predict(eval1)
print(accuracy_score(target, predictions))
print(confusion_matrix(target, predictions))
print(classification_report(target, predictions))


# Random Forest

# In[ ]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train, y_train)


# In[ ]:


# Actual class predictions
rf_predictions = model.predict(X_test)
# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[ ]:



from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[ ]:


y_pred = classifier.predict(eval1)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(target, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(target, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(target, y_pred)
print("Accuracy:",result2)

