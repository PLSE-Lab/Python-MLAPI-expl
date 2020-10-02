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
# print(os.listdir("../input/avani24"))
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# # IMPORT NECESSARY LIBRARIES TO READING THE DATA

# In[ ]:


import pandas as pd # Data processing
import numpy as np # For Linear Algebra Calculation


# In[ ]:


heart_disease = pd.read_csv('../input/heart/heart.csv')
heart_disease.tail()


# It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
# 
# 1.age: The person's age in years
# 
# 2.sex: The person's sex (1 = male, 0 = female)
# 
# 3.cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# 4.trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# 5.chol: The person's cholesterol measurement in mg/dl
# 
# 6.fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 7.restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# 8.thalach: The person's maximum heart rate achieved
# 
# 9.exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 10.oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# 11.slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# 12.ca: The number of major vessels (0-3)
# 
# 13.thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 14.target: Heart disease (0 = no, 1 = yes)
# 

# In[ ]:


# Find how many variables and objects in the data set
heart_disease.shape


# In[ ]:


# view the type of data in the data set
heart_disease.info()


# # Change the Object names as meaningfull

# In[ ]:


heart_disease= heart_disease.rename(columns= {'cp': 'chest_pain_type' , 'trestbps': 'resting_blood_pressure' , 'chol' : 'cholesterol',
                                             'fbs': 'fasting_blood_sugar' , 'restecg' : 'rest_ecg' ,'thalach' : 'max_heart_rate_achieved',
                                             'exang' : 'exercise_induced_angina' , 'oldpeak' : 'st_depression' , 'slope' : 'st_slope',
                                             'ca' : 'num_major_vessels' , 'thal' : 'thalassemia'})


# In[ ]:


# View the first 10 rows in data set
heart_disease.head(10)


# In[ ]:


# View the last 10 rows in the data set
heart_disease.tail(10)


# # Check if any missing values in the data

# In[ ]:


heart_disease.isnull().sum()


# There is no  NAN / NA values in the given data set

# # Data Cleaning on Categorical Data

# In[ ]:


# Convert Sex Column data
heart_disease['sex'][heart_disease['sex'] == 0] = 'Female'
heart_disease['sex'][heart_disease['sex'] == 1] = 'Male'


# In[ ]:


# Convert Chest pain type column data
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 0] = 'typical angina'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 1] = 'atypical angina'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 2] = 'non-anginal pain'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 3] = 'asymptomatic'


# In[ ]:


# Convert Fast Blood sugar column
heart_disease['fasting_blood_sugar'][heart_disease['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
heart_disease['fasting_blood_sugar'][heart_disease['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'


# In[ ]:


# Convert rest_ecg column data
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 0] = 'normal'
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 1] = 'ST-T wave abnormality'
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 2] = 'left ventricular hypertrophy'


# In[ ]:


# Convert exercise_included_angina
heart_disease['exercise_induced_angina'][heart_disease['exercise_induced_angina'] == 0] = 'no'
heart_disease['exercise_induced_angina'][heart_disease['exercise_induced_angina'] == 1] = 'yes'


# In[ ]:


# Convert solpe column data
heart_disease['st_slope'][heart_disease['st_slope'] == 1] = 'upsloping'
heart_disease['st_slope'][heart_disease['st_slope'] == 2] = 'flat'
heart_disease['st_slope'][heart_disease['st_slope'] == 3] = 'downsloping'


# In[ ]:


# convert Thalassemia column data
heart_disease['thalassemia'][heart_disease['thalassemia'] == 1] = 'normal'
heart_disease['thalassemia'][heart_disease['thalassemia'] == 2] = 'fixed defect'
heart_disease['thalassemia'][heart_disease['thalassemia'] == 3] = 'reversable defect'


# In[ ]:


# View the data set after changing it to Categorical
heart_disease.head(10)


# # Exploaratory Data Analysis

# In[ ]:


heart_disease.describe().transpose()


# In[ ]:


# Calaculte on individual column count -Sex
heart_disease['sex'].value_counts()


# In[ ]:


# Calaculte on individual column count -chest_pain_type
heart_disease['chest_pain_type'].value_counts()


# In[ ]:


# Calculate on individual column count - fasting_blood_sugar
heart_disease['fasting_blood_sugar'].value_counts()


# In[ ]:


# Calculate on individual column count - rest_ecg
heart_disease['rest_ecg'].value_counts()


# In[ ]:


# Calculate on individual column count -exercise_induced_angina
heart_disease['exercise_induced_angina'].value_counts()


# In[ ]:


# Calculate on individual column count -st_slope
heart_disease['st_slope'].value_counts()


# In[ ]:


# Calculate on individual column count -  thalassemia
heart_disease['thalassemia'].value_counts()


# # Skewness and Kurtosis

# In[ ]:


# Import Libraries
from scipy.stats import skew , kurtosis


# In[ ]:


# Calculate Skewnes and Kurtosis on individual columns -Sex
print("skewness of the age" , skew(heart_disease['age']))
print("Kurtosis of Age ", kurtosis(heart_disease['age']))


# In[ ]:


# Calculate Skewnes and Kurtosis on individual columns - resting_blood_pressure
print("skewness of the resting_blood_pressure" , skew(heart_disease['resting_blood_pressure']))
print("Kurtosis of resting_blood_pressure ", kurtosis(heart_disease['resting_blood_pressure']))


# In[ ]:


# Calculate Skewnes and Kurtosis on individual columns - cholesterol
print("skewness of the cholesterol" , skew(heart_disease['cholesterol']))
print("Kurtosis of cholesterol ", kurtosis(heart_disease['cholesterol']))


# In[ ]:


# Calculate Skewnes and Kurtosis on individual columns - max_heart_rate_achieved
print("skewness of the max_heart_rate_achieved" , skew(heart_disease['max_heart_rate_achieved']))
print("Kurtosis of max_heart_rate_achieved ", kurtosis(heart_disease['max_heart_rate_achieved']))


# In[ ]:


# Calculate Skewnes and Kurtosis on individual columns - st_depression
print("skewness of the st_depression" , skew(heart_disease['st_depression']))
print("Kurtosis of st_depression ", kurtosis(heart_disease['st_depression']))


# # Various Graphical Visualizations

# # Univariate Analysis

# In[ ]:


# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns


# # 1. People getting Heart disease of Aged persons

# In[ ]:


fig,ax = plt.subplots(figsize=(5,5))
ax = sns.countplot(heart_disease['age'])
plt.show()


# # 2. Most of Male people getting Heart Disease

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['sex'])
plt.show()


# # 3. Chest Pain  type is more of Typical Angina

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['chest_pain_type'])
plt.show()


# # 4. Target

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['target'])
plt.show()


# # 4. Excercise againist Heart Disease

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['exercise_induced_angina'])
plt.show()


# # 6. ECG

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['rest_ecg'])
plt.show()


# # 7. Slope of Heart Disease

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['st_slope'])
plt.show()


# # 8. Thalassemia

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['thalassemia'])
plt.show()


# # Bivariate Analysis

# In[ ]:


sns.distplot(heart_disease['age'])


# In[ ]:


heart_disease.head()


# In[ ]:


sns.distplot(heart_disease['cholesterol'])


# In[ ]:


sns.distplot(heart_disease['resting_blood_pressure'])


# In[ ]:


sns.distplot(heart_disease['max_heart_rate_achieved'])


# In[ ]:



sns.distplot(heart_disease['st_depression'])


# In[ ]:



sns.distplot(heart_disease['num_major_vessels'])


# # 1. Age Vs Sex

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='sex',y='age',data=heart_disease)
plt.show()


# # 2. Age Vs Heart Rate Type

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='max_heart_rate_achieved',data=heart_disease)
plt.show()


# # 3. Age Vs Heart Rate

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='max_heart_rate_achieved',data=heart_disease)
plt.show()


# # 4. Age Vs target

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='target',data=heart_disease)
plt.show()


# # 5. Age Vs Cholestrol

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='cholesterol',data=heart_disease)
plt.show()


# # Distribution of the Target

# In[ ]:


sns.distplot(heart_disease['target'])


# In[ ]:


pd.crosstab(heart_disease.age,heart_disease.target).plot(kind="bar",figsize=(25,8),color=['gold','brown' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


pd.crosstab(heart_disease.sex,heart_disease.target).plot(kind="bar",figsize=(10,5),color=['cyan','coral' ])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# # Pairplot

# In[ ]:


sns.pairplot(data=heart_disease)


# # Correlation Diagram

# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(heart_disease.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()


# # Engineering Featuring

# In[ ]:


heart_disease.groupby('chest_pain_type', as_index=False)['target'].mean()


# In[ ]:


heart_disease.groupby('st_slope',as_index=False)['target'].mean()


# In[ ]:


heart_disease.groupby('thalassemia',as_index=False)['target'].mean()


# In[ ]:


heart_disease.groupby('target').mean()


# # Convert to categorical data using Dummy

# In[ ]:


# Convert the data into categorical data type
heart_disease.chest_pain_type = heart_disease.chest_pain_type.astype("category")
heart_disease.exercise_induced_angina = heart_disease.exercise_induced_angina.astype("category")
heart_disease.fasting_blood_sugar = heart_disease.fasting_blood_sugar.astype("category")
heart_disease.rest_ecg = heart_disease.rest_ecg.astype("category")
heart_disease.sex = heart_disease.sex.astype("category")
heart_disease.st_slope = heart_disease.st_slope.astype("category")
heart_disease.thalassemia = heart_disease.thalassemia.astype("category")


# In[ ]:


# Dummy values
heart_disease1 = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease1)


# In[ ]:


heart_disease1.head()


# # Normalize / Scale the Data

# In[ ]:


# Import Libraries
from sklearn.preprocessing import scale
scale(heart_disease1)


# There is -Ve values in normalized data ,hence we can use exponential of the scaled data

# In[ ]:


np.exp(scale(heart_disease1))


# # Divide the data as input and output

# In[ ]:


x = heart_disease1.drop(['target'], axis = 1)
y = heart_disease1.target.values


# In[ ]:


# Input values
x


# In[ ]:


# Output Values
y


# # Split the data as Training and Testing

# In[ ]:


# Import Libraries
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80)


# # Build Machine Learning Models

# # 1 . Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# In[ ]:


# Fit the model
logmodel.fit(x_train,y_train)


# In[ ]:


# Predict the model
LR_pred = logmodel.predict(x_test)
LR_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(LR_pred,y_test))


# In[ ]:


# Accuracy
from sklearn.metrics import accuracy_score
LR_accuracy = accuracy_score(LR_pred,y_test)
LR_accuracy


# # Logistic Regression Accuracy Score is : 87%

# # 2. K-Nearest Neighbor (KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier


# In[ ]:


# Fit the Model
classifier.fit(x_train,y_train)


# In[ ]:


# Predict the Model
knn_pred = classifier.predict(x_test)
knn_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(knn_pred,y_test))


# In[ ]:


# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_knn=accuracy_score(knn_pred,y_test)
accuracy_knn


# # KNN Accuracy Score is :65 %

# # 3. Naive Bayes Classifier (NBC)

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2


# In[ ]:


# Fit the model
classifier2.fit(x_train,y_train)


# In[ ]:


# Predict the model
NBC_pred = classifier2.predict(x_test)
NBC_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(NBC_pred,y_test))


# In[ ]:


# Accuracy
from sklearn.metrics import accuracy_score
NBC_accuracy = accuracy_score(NBC_pred,y_test)
NBC_accuracy


# # NBC Accuracy Score is : 80%

# # 4. Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier1


# In[ ]:


# Fit the model
classifier1.fit(x_train,y_train)


# In[ ]:


# Predict the model
DT_pred = classifier1.predict(x_test)
DT_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(DT_pred,y_test))


# In[ ]:


# Accuracy
from sklearn.metrics import accuracy_score
accuracy_DT = accuracy_score(DT_pred,y_test)
accuracy_DT


# # Decision Tree Classifier Accuracy score is : 74%

# # 5. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(criterion='entropy',random_state=0)
classifier3


# In[ ]:


# Fit the model
classifier3.fit(x_train,y_train)


# In[ ]:


# Predict the model
RF_pred = classifier3.predict(x_test)
RF_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(RF_pred,y_test))


# In[ ]:


# Accuracy
from sklearn.metrics import accuracy_score
accuracy_RF = accuracy_score(RF_pred,y_test)
accuracy_RF


# # Random Forest Classifier Accuracy score is : 82%

# # 6. Support Vector Machine (SVM)

# In[ ]:


from sklearn.svm import SVC
classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4


# In[ ]:


# Fit the model
classifier4.fit(x_train,y_train)


# In[ ]:


# Predict the model
SVC_pred = classifier4.predict(x_test)
SVC_pred


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(SVC_pred,y_test))


# In[ ]:


# Accuracy
from sklearn.metrics import accuracy_score
accuracy_SVC = accuracy_score(SVC_pred,y_test)
accuracy_SVC


# # SVM Accuracy Score is : 77%

# # 7. GridSearchCV Algorithm

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier()


# In[ ]:


parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
classifier5 = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
classifier5


# In[ ]:


# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)


# In[ ]:



# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train, y_train)


# In[ ]:


# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_


# In[ ]:


# Fit the best algorithm to the data. 
clf.fit(x_train, y_train)


# In[ ]:


predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))


# # GridSearchCV Accuracy Score is : 81 %

# # Accuracy Scores for All ML Models:
# # 1. K-Nearest Neighbor : 65 %
# # 2. Decision Trees : 74 %
# # 3. Support Vector Machine : 77 %
# # 4. Naive Bayes Classifier : 80 %
# # 5. Random Forest : 82 %
# # 6. Logistic Regression : 87 %
# # 7. GridSearch CV : 81 %

# # We can use Logistic Regression model to predict the heart disease.

# In[ ]:




