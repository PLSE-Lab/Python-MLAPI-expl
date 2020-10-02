#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_data = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv")
test_data = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv")


# In[ ]:


train_data.head(10)


# In[ ]:


train_data.columns


# In[ ]:


train_data.dtypes


# In[ ]:


train_data.describe()


# In[ ]:


train_data['ApplicantIncome'].plot.box()


# In[ ]:


train_data.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle(" ")


# In[ ]:


train_data.boxplot(column='ApplicantIncome', by = 'Self_Employed')
plt.suptitle(" ")


# In[ ]:


train = train_data


# In[ ]:


train.isnull().sum()


# In[ ]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace= True)
train['Married'].fillna(train['Married'].mode()[0], inplace= True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace= True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace= True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace= True)


# In[ ]:


train.isnull().sum()


# In[ ]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace= True)


# In[ ]:


train['Loan_Amount_Term'].value_counts()


# In[ ]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace= True)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.columns


# In[ ]:


train = train.drop('Loan_ID', axis=1)


# In[ ]:


X = train.drop('Loan_Status', axis=1)
y = train.Loan_Status


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Finding the best model for our dataset using Cross Validation

# In[ ]:


classifier = []
classifier.append(("LogisticReg", LogisticRegression(solver='liblinear', multi_class='ovr')))
classifier.append(("CART", DecisionTreeClassifier(criterion = 'entropy')))
classifier.append(("KNN", KNeighborsClassifier()))
classifier.append(("KernelSVM", SVC(gamma='auto')))
classifier.append(("NaiveBayes", GaussianNB()))
classifier.append(("RandomForest", RandomForestClassifier()))


# In[ ]:


seed = 0
results = []
names = []
for name, model in classifier:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# As per the output, Logistic Regression seems to be the best with 80% accuracy and 5% variance, so lets apply it to the dataset

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[ ]:


y_pred = logreg.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


accuracy_score = accuracy_score(y_test, y_pred)
print(accuracy_score)


# In[ ]:


report = classification_report(y_test, y_pred)
print(report)


# Now, its time to apply our model on the test_data and find the predicted values

# In[ ]:


test_data.shape


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test = test_data.drop('Loan_ID', axis=1)


# In[ ]:


test['Gender'].fillna(test['Gender'].mode()[0], inplace= True)
test['Married'].fillna(test['Married'].mode()[0], inplace= True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace= True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace= True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace= True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace= True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace= True)


# In[ ]:


test.isnull().sum()


# In[ ]:


test = pd.get_dummies(test)


# In[ ]:


test_pred = logreg.predict(test)


# In[ ]:


Submission = pd.DataFrame()
Submission['Loan_ID'] = test_data['Loan_ID']
Submission['Loan_Status'] = test_pred.reshape((test_pred.shape[0]))


# In[ ]:


Submission.head(10)


# In[ ]:


#Submission.to_csv('sample_submission.csv', index=False)


# In[ ]:




