#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries

# In[ ]:


# For basic data manipulation
import numpy as np # linear algebra
import pandas as pd  # data preprocessing


# In[ ]:


#For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# Reading Data

# In[ ]:


# reading the data
data = pd.read_csv('../input/data.csv')


# In[ ]:


# Shapre of data
data.shape


# In[ ]:


# printing few lines
data.head(10) # first 10


# In[ ]:


# describing the data
data.describe()


# In[ ]:


# getting the info
data.info()


# In[ ]:


# As we can see the last column contains all null values, so we can remove it
data = data.drop('Unnamed: 32', axis=1)


# In[ ]:


data.shape


# In[ ]:


# check if dataset contains any null value
data.isnull().sum().sum()


# In[ ]:


# the id values are all unique and won't be required in computation
# so we can remove it

data = data.drop('id', axis = 1)


# Data Visualizations

# In[ ]:


# Some Data Visualization for better understanding
# Here B refers to Benign which means cells are safe from cancer and M means Malignant 
# which means cells are poisonous and can lead to cancer


# 1. Bar Chart
d = data.diagnosis
ax = sns.countplot(d, label='Count')
B, M = d.value_counts()
print('Number of Benign:', B)
print('Number of Malignant:', M)


# In[ ]:


# 2. Percentage 
# plotting a pie chart 

labels = 'Benign', 'Malignant'
colors = ['Red', 'Green']
explode = [0, 0.1]
plt.rcParams['figure.figsize'] = (6,6)
plt.pie(d.value_counts(), colors = colors, labels = labels,explode = explode, autopct = '%.1f%%')
plt.title('Cell Types', fontsize = 18)
plt.show()


# In[ ]:


# Correlation Matrix

correlation = data.corr()

# tick labels
matrix_cols = correlation.columns.tolist()

# convert to array
corr_array = np.array(correlation)


# In[ ]:


# Plotting Correlation heatmap

f, ax = plt.subplots(figsize =(20, 15))
sns.heatmap(correlation, mask=np.zeros_like(correlation, dtype=np.bool), cmap=sns.diverging_palette(220,20,as_cmap=True), square=True, ax=ax)


# Pairplot

# In[ ]:


sns.pairplot(data)


# Box Plots

# In[ ]:


# Box Plots for mean

# box plots are useful for seeing outliers

plt.rcParams['figure.figsize'] = (18,16)

plt.subplot(2,2,1)
sns.boxplot(x = 'diagnosis', y='radius_mean', data=data, palette='Blues')
plt.title('Diagnosis vs radius_mean', fontsize=16)


plt.subplot(2,2,2)
sns.boxplot(x = 'diagnosis', y='texture_mean', data=data, palette='bright')
plt.title('Diagnosis vs texture_mean', fontsize=16)


plt.subplot(2,2,3)
sns.boxplot(x = 'diagnosis', y='perimeter_mean', data=data, palette='spring')
plt.title('Diagnosis vs perimeter_mean', fontsize=16)


plt.subplot(2,2,4)
sns.boxplot(x = 'diagnosis', y='area_mean', data=data, palette='deep')
plt.title('Diagnosis vs area_mean', fontsize=16)


plt.show()


# Boxen Plots

# In[ ]:


# Boxen Plots for Smoothness

# box plots are useful for seeing outliers

plt.rcParams['figure.figsize'] = (18,16)

plt.subplot(2,2,1)
sns.boxenplot(x = 'diagnosis', y='smoothness_mean', data=data, palette='Blues')
plt.title('Diagnosis vs smoothness_mean', fontsize=16)


plt.subplot(2,2,2)
sns.boxenplot(x = 'diagnosis', y='compactness_mean', data=data, palette='bright')
plt.title('Diagnosis vs compactness_mean', fontsize=16)


plt.subplot(2,2,3)
sns.boxenplot(x = 'diagnosis', y='concavity_mean', data=data, palette='spring')
plt.title('Diagnosis vs concavity_mean', fontsize=16)


plt.subplot(2,2,4)
sns.boxenplot(x = 'diagnosis', y='concave points_mean', data=data, palette='deep')
plt.title('Diagnosis vs concave points_mean', fontsize=16)

plt.show()


# Strip Plots

# In[ ]:


# Strip Plots

plt.rcParams['figure.figsize'] = (18,16)

plt.subplot(2,2,1)
sns.stripplot(x = 'diagnosis', y='concavity_se', data=data, palette='Blues')
plt.title('Diagnosis vs concavity_se', fontsize=16)


plt.rcParams['figure.figsize'] = (18,16)
plt.subplot(2,2,2)
sns.stripplot(x = 'diagnosis', y='concave points_se', data=data, palette='bright')
plt.title('Diagnosis vs concave points_se', fontsize=16)


plt.rcParams['figure.figsize'] = (18,16)
plt.subplot(2,2,3)
sns.stripplot(x = 'diagnosis', y='symmetry_se', data=data, palette='spring')
plt.title('Diagnosis vs symmetry_se', fontsize=16)


plt.rcParams['figure.figsize'] = (18,16)
plt.subplot(2,2,4)
sns.stripplot(x = 'diagnosis', y='fractal_dimension_se', data=data, palette='deep')
plt.title('Diagnosis vs fractal_dimension_se', fontsize=16)

plt.show()


# Data Preprocessing

# In[ ]:


# Label encoding for dependent variable

# importing label encoder

from sklearn.preprocessing import LabelEncoder

# performing label encoding

le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])


# In[ ]:


data['diagnosis'].value_counts()


# In[ ]:


# splitting the dependent and independent variables from the dataset

x = data.iloc[:, 1:]
y = data.iloc[:, 0]


# In[ ]:


print(x.shape)
print(y.shape)


# In[ ]:


# splitting the dataset into training and testing dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 8)


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


# standard Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
                   
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# Modelling

# Decision Tree

# In[ ]:


# importing libraries for calculating prediction scores

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score


# In[ ]:


# Logistic Regression

from sklearn.tree import DecisionTreeClassifier
# create the model
model = DecisionTreeClassifier()

# feed the training data into model
model.fit(x_train, y_train)

# predict the test result 
y_pred = model.predict(x_test)

# Calculating the accuracy
print('Training accuracy: ', model.score(x_train, y_train))
print('Test Accuracy: ', model.score(x_test, y_test))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix
plt.rcParams['figure.figsize']= (5, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# In[ ]:


# Feature importance of decision tree

features = data.columns
imp = model.feature_importances_
indices = np.argsort(imp)

plt.rcParams['figure.figsize']=(15,15)
plt.barh(range(len(indices)), imp[indices])
plt.yticks(range(len(indices)), features[indices])
plt.title('Feature importance for Decision Tree', fontsize=25)
plt.grid()
plt.tight_layout()
plt.show()


# Random Forest Classifier

# In[ ]:


# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# creating a model
model = RandomForestClassifier()

# feeding the training data
model.fit(x_train, y_train)

# predcit the test results
y_pred = model.predict(x_test)

# Calculating the accuracy
print('Training accuracy: ', model.score(x_train, y_train))
print('Test accuracy: ', model.score(x_test, y_test))

# Classification report
cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix
plt.rcParams['figure.figsize'] = (5, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')


# In[ ]:


# Feature importance for random forest
features = data.columns
importance = model.feature_importances_
indices = np.argsort(importance)

plt.rcParams['figure.figsize'] = (15, 15)
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), features[indices])
plt.title('Feature Importance for Random Forest', fontsize = 30)
plt.grid()
plt.tight_layout()
plt.show()


# Support Vector Classifier

# In[ ]:


# Support Vector classfier

from sklearn.svm import SVC

# Create a model
model = SVC()

# feed the training data
model.fit(x_train, y_train)

# predicting test results
y_pred = model.predict(x_test)

# Claculating the accuracy
print('Training accuracy: ', model.score(x_train, y_train))
print('Test accuracy: ', model.score(x_test, y_test))

# Classification repot
cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix
plt.rcParams['figure.figsize'] = (5,5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap = 'Greens')


# Support Vector Classifiers show the best results.

# Thanks for reading the kernel.
