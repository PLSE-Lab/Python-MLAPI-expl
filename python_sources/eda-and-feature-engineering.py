#!/usr/bin/env python
# coding: utf-8

# # Diagnosing Heart Disease
# 

# ## Contents

# > 1. Introduction
# 2. EDA
# 3. Feature Engineering
# 4. The Model
# 5. Evaluation
# 6. Conclusion

# ## Introduction

# This dataset gives a number of variables along with a target condition of having or not having heart disease. Below, the data is first used in a simple Decision Tree and Random Forest models, and then the model is evaluated using different techniques.

# ### Loading the libraries

# In[120]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os

from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import model_selection


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.metrics import roc_curve, auc #for model evaluation


# In[65]:


df = pd.read_csv('../input/heart.csv')


# ## EDA

# View the top 5 rows

# In[60]:


df.head()


# In[61]:


#See the columns
df.columns


# Understanding the features:
# 
# * age: The person's age in years
# * sex: The person's sex (1 = male, 0 = female)
# * cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# * trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# * chol: The person's cholesterol measurement in mg/dl
# * fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# * restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# * thalach: The person's maximum heart rate achieved
# * exang: Exercise induced angina (1 = yes; 0 = no)
# * oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# * slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# * ca: The number of major vessels (0-3)
# * thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# * target: Heart disease (0 = no, 1 = yes)

# In[3]:


df.shape


# There are 383 examples and 14 features

# In[5]:


#what is the target variable
df['target'].unique()


# target variable says if there is a presence of cancer or not.

# In[6]:


#lets see how many nulls do we have
print(df.isnull().sum())


# We do not have any null values in the dataset

# In[7]:


#Data types of features
df.dtypes


# All features are numerical

# In[8]:


df.describe()


# sex, fbs, and exang are binary in nature

# Lets explore target variable with different features

# In[9]:


#count of target variables
print(df.groupby('target')['target'].count())
sns.countplot("target",data = df)


# The classes are very much balanced

# In[10]:


# How age and heart disease relates
plt.figure(figsize = (19, 8))
sns.countplot('age', data=df,hue='target');
plt.title('Age v/s Cancer');


# In[11]:


#Lets see the % of different age groups that have heart disease to understand better
age_target = df[df['target'] ==1].groupby('age')['target'].sum()
bins = [10, 20, 30, 40,50,60,70,80,90]
binned_age = pd.cut(age_target.values, bins=bins).value_counts()
plt.figure(figsize=(16, 8))
plt.hist(df.age,bins = 10)
plt.xlabel("Age")
plt.ylabel("Count")


# In[13]:


age_target.values


# In[14]:


#lets see how heart disease is related with Sex
sns.countplot('sex',data = df, hue='target')


# In[15]:


#Count of each sex
print(df.groupby('sex')['sex'].count())
sns.countplot("sex",data = df)


# In[16]:


#  % of each sex having heart disease
df[df['target'] ==1].sex.value_counts()/df.sex.value_counts()


# In[18]:


df[df['target'] ==1].sex.value_counts()


# 93% of Sex type 0 and 72% of Sex type 1 have positive results

# Lets see the distribution of other features

# In[52]:


features = ['trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','cp']


# In[58]:


f, axes = plt.subplots(3, 5, figsize=(20, 20))
plt.suptitle('Distribution plots')
i=0
for feature in features:
        sns.distplot(df[feature],hist=False,label=feature,ax=axes[i // 5][i % 5]);
        i+=1


# trestbps, chol, thalach has a higher range of values

# In[59]:


#lets see if there is any correaltion amoungst them
plt.figure(figsize=(15, 8))
sns.heatmap(df[features].corr(),annot = True)


# There is not much correlation between the features!

# ## Feature Engineering

# Normalizing trestbps, chol, and thalach 

# In[66]:


#trestbps
a, b = 1, 10
m, n = df.trestbps.min(), df.trestbps.max()
df['trestbps'] = (df.trestbps - m) / (n - m) * (b - a) + a


# In[67]:


#trestbps
a, b = 1, 10
m, n = df.chol.min(), df.chol.max()
df['chol'] = (df.chol - m) / (n - m) * (b - a) + a


# In[68]:


#trestbps
a, b = 1, 10
m, n = df.thalach.min(), df.thalach.max()
df['thalach'] = (df.thalach - m) / (n - m) * (b - a) + a


# ## Model preparation
# 
# 

# In[69]:


X = df.loc[:, df.columns != 'target'].values
y = df['target'].values


# #Splitting the data into train and test

# In[107]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 


# Training the data with Random Forest using cross validation

# In[84]:


model = ensemble.RandomForestClassifier(n_estimators=800)

# create the ensemble model
seed = 450
kfold = model_selection.StratifiedKFold(random_state=seed)
results = model_selection.cross_val_score(model, X_train, 
                    y_train, cv=10)
for i in range(len(results)):
    print("Fold", i+1, "score: ", results[i])
print("Cross-validation score average on original data: ", results.mean())


# Lets see the features importance

# In[76]:


df = df.loc[:, df.columns != 'target']


# In[99]:


model.fit(X, y)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(25, 12))                                                        

feature_importances = pd.DataFrame(importances,index = df.columns, columns = ['importance'])
feature_importances.sort_values('importance', ascending = False).plot(kind = 'bar',
                        figsize = (35,8), color = 'r', yerr=std[indices], align = 'center')
plt.xticks(rotation=90)
plt.xlabel("features",fontsize=16)
plt.show()


# Predicting test data

# In[113]:


model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# ## Evaluating the model

# Accruacy

# In[114]:


ac=accuracy_score(y_test,y_pred)
print("Accuracy ", ac)


# Lets see the confusion matrix along with recall and precision

# In[116]:


rc=recall_score(y_test, y_pred, average='macro')
print("recall score ",rc)

pc=precision_score(y_test, y_pred, average='macro') 

print("precision score ",pc)


# In[118]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In medical industry False Positives are accpetable but not False Negatives.

#  Let's also check with a Receiver Operator Curve (ROC),

# In[121]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# ## Conclusion

# This is a small dataset. I tried to first visualize the different features and then perform feature engineering.
# We should ensure that there are no false negatives as we are dealing with the medical data.
# 
# I will also use Neural Networks as well to see if it performs any better.

# In[101]:


#pruned=[]
#for i in range(len(feature_importances)):
 #   if(feature_importances.importance[i] < 0.04):
  #      pruned.append(feature_importances.index[i])

#len(pruned)


# In[ ]:




