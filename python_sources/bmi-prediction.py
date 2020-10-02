#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


BMI = pd.read_csv('../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv')


# In[ ]:


BMI.tail()


# In[ ]:


gender_label = LabelEncoder()
BMI['Gender'] = gender_label.fit_transform(BMI['Gender'])


# In[ ]:


bins = (-1,0,1,2,3,4,5)
health_status = ['Extremely Underweight','Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese']
BMI['Index'] = pd.cut(BMI['Index'], bins = bins, labels = health_status)
BMI['Index'].value_counts()


# In[ ]:


sns.countplot(BMI['Index'])


# In[ ]:


BMI.info()


# In[ ]:


BMI['Index']


# In[ ]:


BMI.tail()


# In[ ]:


X = BMI.drop('Index', axis = 1)
y = BMI['Index']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# # SVM Classifier

# In[ ]:


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# # Neural Network

# In[ ]:


mlpc = MLPClassifier(hidden_layer_sizes=(3,3,3), max_iter = 500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# # Test

# In[ ]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)
cm


# In[ ]:


Xnew = [[1,175,35]] #Enter New Value: [Gender, Height, Weight]
Xnew = sc.transform(Xnew)
ynew = rfc.predict(Xnew)
ynew

