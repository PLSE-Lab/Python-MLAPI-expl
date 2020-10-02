#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import Libraries

# In[ ]:


import pandas as pd 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# Read datasets

# In[ ]:


df = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')
df.head()


# # Data Analysing and Cleaning

# In[ ]:


print('Shape:', df.shape)
print(df.columns)


# In[ ]:


df.dtypes


# In[ ]:


df.describe().T


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(4,4))
sns.countplot(x='Outcome', data=df)
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.show()


# In[ ]:


df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure',
                                                                                'SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)


# # Split to test and validation sets

# In[ ]:


X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))


# Feature Scaling

# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Using cross_val_score for gaining average accuracy

# In[ ]:


rf= cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0), X_train, y_train, cv=5)
print("Mean Accuracy: %f" % (rf.mean()*100))


# # Random Forest Model

# In[ ]:


classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)


# # Results

# In[ ]:


print('Train accuracy :', (classifier.score(X_train, y_train))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_train, classifier.predict(X_train)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_train, classifier.predict(X_train)))


# In[ ]:


print('Test accuracy :', (classifier.score(X_test, y_test))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_test, classifier.predict(X_test)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, classifier.predict(X_test)))


# Save model

# In[ ]:


file = 'diab_model.pkl'
pickle.dump(classifier, open(file, 'wb'))


# # Testing the model

# In[ ]:



def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    preg = int(Pregnancies)
    glucose = float(Glucose)
    bp = float(BloodPressure)
    st = float(SkinThickness)
    insulin = float(Insulin)
    bmi = float(BMI)
    dpf = float(DiabetesPedigreeFunction)
    age = int(Age)

    x = [[preg, glucose, bp, st, insulin, bmi, dpf, age]]
    x = sc.transform(x)

    return classifier.predict(x)


# Input Order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age

# Prediction 1

# In[ ]:


prediction = predict_diabetes(5, 81, 73, 24, 76, 30.1, 0.567, 23)[0]
if prediction:
  print('You have diabetes.')
else:
  print("You don't have diabetes.")


# Prediction 2

# In[ ]:


prediction = predict_diabetes(2, 120, 88, 23, 192, 35.5, 0.408, 38)[0]
if prediction:
  print('You have diabetes.')
else:
  print("You don't have diabetes.")


# # Conclusion
# 
# We have hence predicted a model with a validation accuracy of 98.75%.
