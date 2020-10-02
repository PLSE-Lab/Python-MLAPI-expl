#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Let's check the dataset
df = pd.read_csv('../input/indian_liver_patient.csv')
df.head()


# > **Data Analysis**

# In[ ]:


#Let's checl the descriptive statistics
df.describe(include = 'all')


# In[ ]:


#Let's check the data types

df.dtypes


# In[ ]:


#Let's check if any data has null values
df.isnull().sum()


# In[ ]:


#Let's check the shape of dataset
df.shape


# In[ ]:


df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].replace(np.nan, 0)


# In[ ]:


#Checking whether all columns are having the string data types
all(isinstance(column, str) for column in df.columns)


# **Data Visualization**

# In[ ]:


#Importing Viuualization libraries using matlpotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Creating correlation matrix
df_corr = df.corr()
#Plot figsize
plt.subplots(figsize=(10, 10))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(df_corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(df_corr.columns)), df_corr.columns);
#Apply yticks
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
#show plot
plt.show()


# In[ ]:


columns = df.columns
columns


# In[ ]:


features = list(map(str, df.columns))
features


# In[ ]:


sns.pairplot(df.select_dtypes(include=[np.number]), dropna=True)


# In[ ]:


#Comparison of patients with and without liver diseases
sns.countplot(data=df, x = 'Dataset', label='Count')

LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[ ]:


sns.countplot(data=df, x = 'Gender', label='Count')

M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[ ]:


sns.factorplot(x="Age", y="Gender", hue="Dataset", data=df);


# In[ ]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[ ]:


sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[ ]:


sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[ ]:


sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=df, kind="reg")


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[ ]:


sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[ ]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")


# In[ ]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# **Machine Leaening**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

num = LabelEncoder()
df['Gender'] = num.fit_transform(df['Gender'].astype('str'))
df.head()


# In[ ]:


features = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
            'Albumin_and_Globulin_Ratio']
X= df[features]
y= df['Dataset']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
y,
random_state=0)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc1= svc.score(X_train, y_train)
svc2 = svc.score(X_test, y_test)
print(svc1, svc2)
print('Classification Report: \n', classification_report(y_test,y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test,y_pred))
from sklearn.model_selection import cross_val_score
print(cross_val_score(svc, X_train, y_train, cv=10))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

clf = LogisticRegression(C=100,penalty="l1").fit(X_train, y_train)
clf_predict = clf.predict(X_test)
clf1 = clf.score(X_train,y_train)
clf2 = clf.score(X_test,y_test)
print('Logistic Regression Training Score: \n', clf1)
print('Logistic Regression Test Score: \n', clf2)
print('Coefficient: \n', clf.coef_)
print('Intercept: \n', clf.intercept_)
print('Accuracy: \n', accuracy_score(y_test,clf_predict))
print('Confusion Matrix: \n', confusion_matrix(y_test,clf_predict))
print('Classification Report: \n', classification_report(y_test,clf_predict))

sns.heatmap(confusion_matrix(y_test,clf_predict),annot=True,fmt="d")


# In[ ]:


from sklearn.model_selection import cross_val_score
print(cross_val_score(clf, X_train, y_train, cv=10))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train,y_train)
gnb_predict = gnb.predict(X_test)
gnb1= gnb.score(X_train,y_train)
gnb2= gnb.score(X_test,y_test)
gnb1,gnb2
print('Confusion Matrix: \n', confusion_matrix(y_test,gnb_predict))
print('Classification Report: \n', classification_report(y_test,gnb_predict))
print(cross_val_score(gnb, X_train, y_train, cv=10))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print('Accuracy: \n', accuracy_score(y_test,clf_predict))
print('Confusion Matrix: \n', confusion_matrix(y_test,clf_predict))
print('Classification Report: \n', classification_report(y_test,clf_predict))
print(cross_val_score(clf, X_train, y_train, cv=10))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest = random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = random_forest.score(X_train, y_train)
random_forest_score_test = random_forest.score(X_test, y_test)
print('Random Forest train Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print(classification_report(y_test,rf_predicted))
from sklearn.model_selection import cross_val_score
print(cross_val_score(random_forest, X_train, y_train, cv=10))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print('Accuracy: \n', accuracy_score(y_test,clf_predict))
print('Confusion Matrix: \n', confusion_matrix(y_test,clf_predict))
print('Classification Report: \n', classification_report(y_test,clf_predict))
print(cross_val_score(clf, X_train, y_train, cv=10))

