#!/usr/bin/env python
# coding: utf-8

# # **Importing Libraries**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# # **Defining arrays**

# In[ ]:


kfoldmean = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
kfoldstd = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
cmval=np.array([0,0,0,0,0,0,0])
positions=np.array([0,1,2,3,4,5,6])
names=["Logistic Regression","KNN","SVM Linear","SVM rbf","Naive Bayes","Decision Tree","Random Forest"]


# # **Importing dataset**

# In[ ]:


dataset = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values


# # **Handling missing values** and replacing it with mean

# In[ ]:


from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=0, strategy="mean")
imputer=imputer.fit(X[:,1:8])
X[:,1:8]=imputer.transform(X[:,1:8])


# # **Finding Multicollinearity**

# In[ ]:


corr = dataset.corr()
plt.figure(figsize=(12, 12))
sb.heatmap(corr, annot=True)


# No serious Multicollinearity found

# # **Splitting dataset**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)


# # **Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Dataset is ready for processing

# # **Logistic Regression, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[0]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[0]=accuracies.mean()
kfoldstd[0]=accuracies.std()


# # **KNN, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[1]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[1]=accuracies.mean()
kfoldstd[1]=accuracies.std()


# # **SVM Linear, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[2]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[2]=accuracies.mean()
kfoldstd[2]=accuracies.std()


# # **SVM RBF Gaussian Kernel, Confusion Matrix, K-Fold**

# In[ ]:


classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[3]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[3]=accuracies.mean()
kfoldstd[3]=accuracies.std()


# # **Naive Bayes, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[4]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[4]=accuracies.mean()
kfoldstd[4]=accuracies.std()


# # **Decision Tree, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[5]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[5]=accuracies.mean()
kfoldstd[5]=accuracies.std()


# # **Random Forest, Confusion Matrix, K-Fold**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='g')

cmval[6]=cm[0][0]+cm[1][1]

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Mean using K-fold:",accuracies.mean())
print("Standard Deviation using K-fold:",accuracies.std())

kfoldmean[6]=accuracies.mean()
kfoldstd[6]=accuracies.std()


# # **Plotting Correct values by Each Model using Confusion Matrix**

# In[ ]:


fig=plt.figure(figsize=(12,12))
plt.barh(names,cmval)
plt.ylabel('Models Used')
plt.xlabel('Correct CM Values')
for index, value in enumerate(cmval):
    plt.text(value, index, str(value))
plt.show()


# # **Plotting K-Fold Mean Values of Each Model**

# In[ ]:


fig=plt.figure(figsize=(12,12))
plt.barh(names,kfoldmean)
plt.ylabel('Models Used')
plt.xlabel('Kfold Mean Values')
for index, value in enumerate(kfoldmean):
    plt.text(value, index, str(value))
plt.show()


# # **Plotting K-Fold Standard Deviation Values of Each Model **

# In[ ]:


fig=plt.figure(figsize=(12,12))
plt.barh(names,kfoldmean)
plt.ylabel('Models Used')
plt.xlabel('Kfold Standard Deviation Values')
for index, value in enumerate(kfoldstd):
    plt.text(value, index, str(value))
plt.show()


# # Summary

# Best Model found was Logistic Regression with the highest K-Fold Score and Least Incorrect Predicted Values using CM.
