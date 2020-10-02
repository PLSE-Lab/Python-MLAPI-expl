#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/adult-census-income/adult.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.isin(['?']).sum()


# In[ ]:


df = df.replace('?', np.NaN)


# In[ ]:


for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df['income'].value_counts()


# In[ ]:


sns.countplot(x='income', data = df)


# In[ ]:


sns.boxplot(y='age',x='income',data=df)


# In[ ]:


sns.boxplot(y='hours.per.week',x='income',data=df)


# In[ ]:


sns.countplot(df['sex'],hue=df['income'])


# In[ ]:


sns.countplot(df['occupation'],hue=df['income'])
plt.xticks(rotation=90)


# In[ ]:


df['income']=df['income'].map({'<=50K': 0, '>50K': 1})


# In[ ]:


sns.barplot(x="education.num",y="income",data=df)


# In[ ]:


df['workclass'].unique()


# In[ ]:


sns.barplot(x="workclass",y="income",data=df)
plt.xticks(rotation=90)


# In[ ]:


df['education'].unique()


# In[ ]:


sns.barplot(x="education",y="income",data=df)
plt.xticks(rotation=90)


# In[ ]:


df['marital.status'].unique()


# In[ ]:


sns.barplot(x="marital.status",y="income",data=df)
plt.xticks(rotation=90)


# In[ ]:


df['relationship'].unique()


# In[ ]:


sns.barplot(x="relationship",y="income",data=df)
plt.xticks(rotation=90)


# In[ ]:


df['native.country'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])


# In[ ]:


df.dtypes


# In[ ]:


df.head()


# In[ ]:


corrmat = df.corr()
plt.figure(figsize=(20,12))
sns.heatmap(corrmat, annot=True, cmap='coolwarm')


# In[ ]:


corrmat['income'].sort_values(ascending = False)


# In[ ]:


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train))
X_test = pd.DataFrame(sc.transform(X_test))


# In[ ]:


X_train.head()


# In[ ]:


l=[]


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Logistic Regression:', acc * 100)
l.append(acc)


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('SVM:', acc * 100)
l.append(acc)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Knn:',acc * 100)
l.append(acc)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Decision Tree:', acc * 100)
l.append(acc)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Naive Bayes:', acc * 100)
l.append(acc)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Random Forest:',acc * 100)
l.append(acc)
print(cm(y_test, y_pred))
print(cr(y_test, y_pred))


# In[ ]:


y_axis=['Logistic Regression',
     'Support Vector Classifier',
        'K-Neighbors Classifier',
      'Decision Tree Classifier',
       'Gaussian Naive Bayes',
      'Random Forest Classifier']
x_axis=l
sns.barplot(x=x_axis,y=y_axis)
plt.xlabel('Accuracy')


# If you find this kernel useful, **PLEASE UPVOTE!!**
