#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.DataFrame(pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv'))
df.head()


# # EDA

# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['Time'])


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['Amount'])


# In[ ]:


sns.countplot(x='Class',data=df)


# In[ ]:


print("Percentage of non-fraud transactions:", round(100*df[df['Class']==0]['Class'].value_counts()/len(df),2))
print("Percentage of fraud transactions:", round(100*df[df['Class']==1]['Class'].value_counts()/len(df),2))


# In[ ]:


sns.stripplot(x='Class', y='Amount',data=df)


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(30, 25))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


sns.scatterplot(x='Amount',y='V2',data=df)


# In[ ]:


sns.scatterplot(x='Amount', y='V5',data=df)


# In[ ]:


sns.scatterplot(x='Amount', y='V7',data=df)


# In[ ]:


sns.scatterplot(x='Amount', y='V20',data=df)


# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(x='Time', y='V3',data=df)


# Observation:
# 1) Dataset is extremely imbalanced.
# 2) Dataset contains large number of outliers in 'Amount' column. 

# We have to scale both 'Time' and 'Amount' columns as the rest of the columns are already scaled. We will use Robust Scaler as 'Amount' contains large number of outliers. 

# In[ ]:


from sklearn.preprocessing import RobustScaler
robsc = RobustScaler()


# In[ ]:


df['scaled_amount'] = robsc.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = robsc.fit_transform(df['Time'].values.reshape(-1,1))


# In[ ]:


df.drop('Amount', inplace=True, axis=1)
df.drop('Time', inplace=True, axis=1)
display(df.head())


# In[ ]:


scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount','scaled_time'], axis=1, inplace=True)

df.insert(0,'scaled_amount', scaled_amount)
df.insert(1,'scaled_time', scaled_time)


# To solve the problem of imbalanced dataset we will scale the data using both the techniques of undersampling and oversampling. But before we do that we will split the data into training and testing set and apply resampling only on the training set. 

# In[ ]:


from sklearn.model_selection import train_test_split

X=df.drop('Class', axis=1)
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=df['Class'])


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "Support Vector Classifier": SVC()
}


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# Undersampling our training data using Tomek Links:

# In[ ]:


from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_tomek, y_tomek= tl.fit_sample(X_train, y_train)


# In[ ]:


from collections import Counter
print('Resampled dataset shape %s' % Counter(y_tomek))


# We will use 6 models to make our predictions. The 6 models are: Logistic Regression, KNeighbors Classifier, Decision Tree Classifier, Gradient Boosting Classifier, Random Forest Classifier and Support Vector Classifier.  

# In[ ]:


for key, classifier in classifiers.items():
    classifier.fit(X_tomek, y_tomek)
    print("Classifiers: ", classifier.__class__.__name__)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# Oversampling our training data using SMOTE: 

# In[ ]:


from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_sm, y_sm= sm.fit_sample(X_train, y_train)


# In[ ]:


for key, classifier in classifiers.items():
    classifier.fit(X_sm, y_sm)
    print("Classifiers: ", classifier.__class__.__name__)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# Conclusion: Random Forest Classifier gave best result out of all the models for  both undersampled and oversampled data. 
