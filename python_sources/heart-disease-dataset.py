#!/usr/bin/env python
# coding: utf-8

# # **Decision Tree Classification VS XGBClassifier**

# This approach towards this problem of analyzing diffrent heart diseases is based on how decision trees can very easily create deep separation in data to find a correlation in the data that represents some type of heart disease. Or if a XGBClassifier is better at learning how to classifie the data in order to predict at a higher accuracy.

# These will be the imports I use throughout this entire kernel:
# 

# In[ ]:


import numpy as np
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Read and output features of the dataset:

# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head()


# This piece of code is here to check if there are any null or incomplete values in the dataset so then we don't train an incomplete model.

# In[ ]:


df.isnull().sum()[df.isnull().sum()>0]


# Since I have determined there are no empty values, I created various representations on how the decision tree would be able to make a decision on a person by various diffrent correlations within the dataset.

# In[ ]:


warnings.filterwarnings(action="ignore")
plt.figure(figsize=[15,15])
plt.subplot(5,5,1)
sns.barplot('sex','age',data=df)
plt.subplot(5,5,2)
sns.barplot('cp','age',data=df)
plt.subplot(5,5,3)
sns.distplot(df[df.target==1].trestbps, color='green', kde=False)
sns.distplot(df[df.target==0].trestbps, color='orange', kde=False)
plt.subplot(5,5,4)
sns.distplot(df[df.target==1].chol, color='green', kde=False)
sns.distplot(df[df.target==0].chol, color='orange', kde=False)
plt.subplot(5,5,5)
sns.barplot('slope','age',data=df)
plt.subplot(5,5,6)
sns.barplot('thal','age',data=df)
plt.subplot(5,5,7)
sns.barplot('ca','age',data=df)
plt.subplot(5,5,8)
sns.distplot(df[df.target==1].age, color='green', kde=False)
sns.distplot(df[df.target==0].age, color='orange', kde=False)
plt.subplot(5,5,9)
sns.distplot(df[df.target==1].oldpeak, color='green', kde=False)
sns.distplot(df[df.target==0].oldpeak, color='orange', kde=False)
plt.subplot(5,5,10)
sns.barplot('fbs','age',data=df)
plt.subplot(5,5,11)
sns.barplot('restecg','age',data=df)
plt.subplot(5,5,12)
sns.distplot(df[df.target==1].thalach, color='green', kde=False)
sns.distplot(df[df.target==0].thalach, color='orange', kde=False)
plt.subplot(5,5,13)
sns.barplot('exang','age',data=df)
plt.subplot(5,5,14)
sns.distplot(df[df.fbs==1].age, color='green', kde=False)
sns.distplot(df[df.fbs==0].age, color='orange', kde=False)
plt.subplot(5,5,15)
sns.distplot(df[df.fbs==1].trestbps, color='green', kde=False)
sns.distplot(df[df.fbs==0].trestbps, color='orange', kde=False)
plt.subplot(5,5,16)
sns.distplot(df[df.fbs==1].chol, color='green', kde=False)
sns.distplot(df[df.fbs==0].chol, color='orange', kde=False)
plt.subplot(5,5,17)
sns.barplot('restecg','fbs',data=df)
plt.subplot(5,5,18)
sns.distplot(df[df.fbs==1].thalach, color='green', kde=False)
sns.distplot(df[df.fbs==0].thalach, color='orange', kde=False)
plt.subplot(5,5,19)
sns.distplot(df[df.fbs==1].thalach, color='green', kde=False)
sns.distplot(df[df.fbs==0].thalach, color='orange', kde=False)
plt.subplot(5,5,20)
sns.distplot(df[df.fbs==1].oldpeak, color='green', kde=False)
sns.distplot(df[df.fbs==0].oldpeak, color='orange', kde=False)
plt.subplot(5,5,21)
sns.distplot(df[df.fbs==1].slope, color='green', kde=False)
sns.distplot(df[df.fbs==0].slope, color='orange', kde=False)
plt.subplot(5,5,21)
sns.barplot('ca','fbs',data=df)
plt.subplot(5,5,22)
sns.barplot('thal','fbs',data=df)


# Next, I split the Dataset based on all the features and the target being with the test size being 20% of the Dataset.

# In[ ]:


df = pd.get_dummies(df)
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X =df[features]
Y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


# Next, I determined the maximum depth the decision tree would best work on based on the cross-validation score.

# In[ ]:


val = []
for k in range(1,21):
    knn = DecisionTreeClassifier(max_depth=k, random_state=0)
    df_score = pd.DataFrame(cross_val_score(knn, X_train, y_train, cv=10))
    val.append(df_score.mean())
df_val = pd.DataFrame(val)
df_val.idxmax()


# After determining the best max depth which had the best accuracy, 5 , I use it to train the Decision to try and predict based on the test data, then take an accuracy score.

# In[ ]:


knn = DecisionTreeClassifier(max_depth=5, random_state=0)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


# After looking at the Accuracy of the Decision tree I tested the XGBClassifier. After training and predicting the accuracy was way above the Decision tree classifier.

# In[ ]:


model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


# This means that the model created based on the XGBClassifier has an 85% accuracy based on this data. This means that XGBClassifiers are better in deeper models than Decision trees.
