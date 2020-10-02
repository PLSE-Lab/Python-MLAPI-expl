#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


stroke= pd.read_csv('/kaggle/input/healthcare-dataset-stroke-data/train_2v.csv')
stroke.head()


# In[ ]:


#checking the shape of our data
stroke.shape


# In[ ]:


stroke.columns


# In[ ]:


stroke= stroke.drop('id', axis=1)


# In[ ]:


#checking the null values
stroke.isnull().sum()


# In[ ]:


stroke[stroke==0].count()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(stroke['stroke'])


# In[ ]:


#filling null values with the mean
stroke['bmi'].fillna(stroke['bmi'].mean(), inplace= True)


# In[ ]:


#filling null values with mode
stroke['smoking_status'].fillna(stroke['smoking_status'].mode()[0], inplace=True)


# In[ ]:


#checking the data
stroke.isnull().sum()


# In[ ]:


stroke.describe()


# In[ ]:


stroke.info()


# In[ ]:


sns.distplot(stroke['avg_glucose_level'], bins=20)


# In[ ]:


sns.distplot(stroke['bmi'], bins=20)


# In[ ]:


sns.distplot(stroke['age'], bins=20)


# In[ ]:


#chances of stroke incraeses with incraese in age
stroke.loc[stroke['stroke'] == 0,
                 'age'].hist(label='No Stroke')
stroke.loc[stroke['stroke'] == 1,
                 'age'].hist(label='Heart Stroke')
plt.xlabel('Age')
plt.ylabel('Heart Stroke')
plt.legend()


# In[ ]:


#chances of stroke more with bmi 20-40

stroke.loc[stroke['stroke'] == 0,
                 'bmi'].hist(label='No Stroke')
stroke.loc[stroke['stroke'] == 1,
                 'bmi'].hist(label='Heart Stroke')
plt.xlabel('BMI')
plt.ylabel('Heart Stroke')
plt.legend()


# In[ ]:


#chances of stroke high with glucose levels in range of 70-100

stroke.loc[stroke['stroke'] == 0,
                 'avg_glucose_level'].hist(label='No Stroke')
stroke.loc[stroke['stroke'] == 1,
                 'avg_glucose_level'].hist(label='Heart Stroke')
plt.xlabel('Glucose Level')
plt.ylabel('Heart Stroke')
plt.legend()


# In[ ]:


#married females have more chances of heart stroke than married males
pd.pivot_table(stroke, index= 'stroke', columns='gender', values='ever_married', aggfunc= 'count')


# In[ ]:


#females with hypertension has more chance of heart stroke than males having hypertension problem
pd.pivot_table(stroke, index= 'stroke', columns='gender', values='hypertension', aggfunc= 'count')


# In[ ]:


#females with heart disease has more chances of stroke
pd.pivot_table(stroke, index= 'stroke', columns='gender', values='heart_disease', aggfunc= 'count')


# In[ ]:


#people having private jobs and has a habit of smoking has more chance of heart stroke 
pd.pivot_table(stroke, index= 'stroke', columns='work_type', values='smoking_status', aggfunc= 'count')


# In[ ]:


#as age incraeses gender does not play any role in heart stroke
sns.scatterplot(x= 'stroke', y='age', hue='gender', sizes= (15,200), data=stroke)
plt.xticks(rotation=90)


# In[ ]:


#can't say that marriage plays a role in heart stroke as people generally marry after the age of 25years
sns.relplot(x= 'stroke', y='age', hue= 'ever_married', sizes= (15,200), data=stroke)
plt.xticks(rotation=90)


# In[ ]:


#with age glucose level increases which increases the chances of stroke
plt.figure(figsize=(28,20))
sns.relplot(x= 'avg_glucose_level', y='age', hue= 'stroke', sizes= (15,200), data=stroke)
plt.xticks(rotation=90)


# In[ ]:


stroke['stroke'].value_counts()


# In[ ]:


#performing label encoding for the dataset
from sklearn import preprocessing 

encoder = preprocessing.LabelEncoder()

for i in stroke.columns:
    if isinstance(stroke[i][0], str):
            stroke[i] = encoder.fit_transform(stroke[i])


# In[ ]:


#standardizing the dataset with Standard Scaler
from sklearn.preprocessing import StandardScaler 
  
scalar = StandardScaler() 
  
scalar.fit(stroke) 
scaled_data = scalar.transform(stroke)


# In[ ]:


#checing the data for first 10 values
stroke.head(10)


# #Preparing the data for the model
# #first creating a base model

# In[ ]:


#dropping the output label
X= stroke.drop('stroke', axis=1)
X.shape


# In[ ]:


y= stroke['stroke']
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3, random_state = 1000)


# In[ ]:


log= LogisticRegression()


# In[ ]:


log.fit(X_train,y_train)


# In[ ]:


log.score(X_train, y_train)


# #this clearly shows that the model is overfit or is only considering the values which is high in number
# #we need to balance the data which could be done in 2 ways
# #either we can undersample the data by dropping the values or oversample using SMOTE
# #three models i.e. Logistic Regression, Decision Tree and Random Forest will be created after undersampling 
# #confusion matrix for each models will be displayed at the end
# 

# In[ ]:


stroke['stroke'].value_counts()


# In[ ]:


#to retain the original data, we craeted a copy of the dataset
stroke_copy= stroke.copy()


# In[ ]:


stroke_copy.head()


# In[ ]:


#creating a list of data values which is more in number
#to make a balance data
li = list(stroke_copy[stroke_copy.stroke == 0].sample(n=41800).index)


# In[ ]:


#dropping the values
stroke_copy = stroke_copy.drop(stroke_copy.index[li])

stroke_copy['stroke'].value_counts()


# In[ ]:


X_drop= stroke_copy.drop('stroke', axis=1)
X_drop.shape


# In[ ]:


y_drop= stroke_copy.stroke
y_drop.shape


# In[ ]:


X_droptr,X_dropts,y_droptr,y_dropts = train_test_split(X_drop,y_drop,test_size=.3, random_state = 1000)


# In[ ]:


#creating a Logistic Model for the new data
log.fit(X_droptr, y_droptr)


# In[ ]:


#the accuracy has dropped
log.score(X_droptr, y_droptr)


# In[ ]:


#predicting the output with Logistic
y_underlog= log.predict(X_dropts)


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, confusion_matrix
print('The accuracy score of the model is:', accuracy_score(y_dropts,y_underlog)*100)
print('The F1 score of the model is:', f1_score(y_dropts, y_underlog)*100)
print('The recall score of the model is:', recall_score(y_dropts, y_underlog)*100)
print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_underlog))
print('The classification report of logistic model is:', classification_report(y_dropts, y_underlog))


# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[ ]:


model.fit(X_droptr, y_droptr)


# In[ ]:


#tuning the model using criterion and max_depth only

from sklearn.model_selection import GridSearchCV
param = {
    'criterion': ['entropy', 'gini'],
    'max_depth' :[2,3,4,5]
}
grid_svc = GridSearchCV(model, param_grid=param, scoring='accuracy', cv=10)


# In[ ]:


grid_svc.fit(X_droptr, y_droptr)


# In[ ]:


grid_svc.best_params_


# In[ ]:


model= tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 3)
model.fit(X_droptr, y_droptr)


# In[ ]:


model.score(X_droptr, y_droptr)


# In[ ]:


y_undersDT= model.predict(X_dropts)


# In[ ]:


print('The accuracy score of the model is:', accuracy_score(y_dropts,y_undersDT)*100)
print('The F1 score of the model is:', f1_score(y_dropts, y_undersDT)*100)
print('The recall score of the model is:', recall_score(y_dropts, y_undersDT)*100)
print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_undersDT))
print('The classification report of base model is:', classification_report(y_dropts, y_undersDT))


# In[ ]:


from sklearn import ensemble
rf= ensemble.RandomForestClassifier()


# In[ ]:


rf.fit(X_droptr, y_droptr)


# In[ ]:


#tuning the model using criterion, n_estimators, bootstrap and max_depth
param = {
    'criterion': ['entropy', 'gini'],
    'n_estimators': [10,20,30,40,50],
    'bootstrap': ['True', 'False'],
    'max_depth': [2,3,4,5]
}
grid_svc = GridSearchCV(rf, param_grid=param, scoring='accuracy', cv=10)


# In[ ]:


grid_svc.fit(X_droptr, y_droptr)


# In[ ]:


grid_svc.best_params_


# In[ ]:


rf= ensemble.RandomForestClassifier(bootstrap= 'True', criterion= 'entropy', max_depth= 4, n_estimators=50)


# In[ ]:


rf.fit(X_droptr, y_droptr)


# In[ ]:


#checking the accuracy score of the Random Forest Model
rf.score(X_droptr, y_droptr)


# In[ ]:


#predicting the values through Random Forest
y_predRF= rf.predict(X_dropts)


# In[ ]:


print('The accuracy score of the model is:', accuracy_score(y_dropts,y_predRF)*100)
print('The F1 score of the model is:', f1_score(y_dropts, y_predRF)*100)
print('The recall score of the model is:', recall_score(y_dropts, y_predRF)*100)
print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_predRF))
print('The classification report of base model is:', classification_report(y_dropts, y_predRF))


# In[ ]:


cm_log= confusion_matrix(y_dropts, y_underlog)
cm_DT= confusion_matrix(y_dropts, y_undersDT)
cm_RF= confusion_matrix(y_dropts, y_predRF)


# In[ ]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes After Undersampling",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_log,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cm_DT,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_RF,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# In[ ]:




