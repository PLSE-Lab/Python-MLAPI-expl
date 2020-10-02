#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#in this program we will be predicting the quality of the wine
#we have uploaded the dataset from https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine
#there are many factors which determine the quality of the wine
#the following dataset consists of different components that make the wine and we will be finding the relation between the quantity of the components and the quality of the wine


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[ ]:


df.head()


# In[ ]:


df["quality"].unique() #unique values of a column


# In[ ]:


#as we can see the quality parameter differentiates the good wine from the bad once
# 8 - best wine
#the following dataset consists of different components that make the wine and we will be finding the relation between the quantity of the components and the quality of the wine


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = df)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = df)


# In[ ]:


#from the above graphs we are able to find the relation between the quantity of the substance and its relation with the quality of the wine


# In[ ]:


#we will now separate the good wines from the bad once
#Lets conside the winne less than 6.5 and the once greater than that good one
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# In[ ]:


label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])
df['quality'].value_counts()


# In[ ]:


sns.countplot(df['quality'])


# In[ ]:


#separating the input and the output features
X = df.drop('quality', axis = 1)
y = df['quality']


# In[ ]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)


# In[ ]:


sc = StandardScaler()


# In[ ]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


#random forest classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# In[ ]:


#Stochastic Gradient Decent Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))


# In[ ]:


#SVC
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))
print(confusion_matrix(y_test, pred_svc))


# In[ ]:




