#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION

# We have a data which classified if patients have heart disease or not according to features in it. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. In addition, we will analyze for this dataset. We will use a wide range of tools for this part. If there's value in there, we'il do it there. Finally, machine learning algorithms are estimated.

# If you find this kernel helpful, Please <font color="red"><b>UPVOTES</b></font>.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


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


# <br>
# 
# <font size=4px>Importing some useful libraries</font>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

get_ipython().run_line_magic('matplotlib', 'inline')


# <br>
# 
# <font size=4px>Loading the data</font>

# In[ ]:


data = pd.read_csv('../input/heart.csv')
data.head(2)


# <br>
# 
# <font size=4px>Dataset Columns (Features)</font>

# * Age (age in years)
# * Sex (1 = male; 0 = female)
# * CP (chest pain type)
# * TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))
# * CHOL (serum cholestoral in mg/dl)
# * FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# * RESTECH (resting electrocardiographic results)
# * THALACH (maximum heart rate achieved)
# * EXANG (exercise induced angina (1 = yes; 0 = no))
# * OLDPEAK (ST depression induced by exercise relative to rest)
# * SLOPE (the slope of the peak exercise ST segment)
# * CA (number of major vessels (0-3) colored by flourosopy)
# * THAL (3 = normal; 6 = fixed defect; 7 = reversable defect)
# * TARGET (1 or 0)

# <br>
# 
# # Data Exploration

# **Concise summary of a Data**

# In[ ]:


data.info()


# <br>
# **Dimensions of the data : **

# In[ ]:


print('Number of rows in the dataset: ',data.shape[0])
print('Number of columns in the dataset: ',data.shape[1])


# <br>
# **Missing values detection :**

# In[ ]:


data.isnull().sum()


# <font color='green'>There are no null values in the dataset</font>

# **Descriptive statistics Generation.**

# In[ ]:


data.describe()


# <br>
# 
# <font color='skyblue' size=3xp>**Checking how the target values depend on various features.**<font>

# **1. Sex**

# In[ ]:


male = len(data[data.sex == 1])
female = len(data[data.sex == 0])
plt.pie(x=[male, female], explode=(0, 0), labels=['Male', 'Female'], autopct='%1.2f%%', shadow=True, startangle=90)
plt.show()


# In[ ]:


sn.countplot('sex',hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **2. CP (chest pain type)**

# In[ ]:


x = [len(data[data['cp'] == 0]),len(data[data['cp'] == 1]), len(data[data['cp'] == 2]), len(data[data['cp'] == 3])]
plt.pie(x, data=data, labels=['CP(1) typical angina', 'CP(2) atypical angina', 'CP(3) non-anginal pain', 'CP(4) asymptomatic'], autopct='%1.2f%%', shadow=True,startangle=90)
plt.show()


# In[ ]:


sn.countplot('cp',hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for chest pain type')
plt.xlabel('Chest pain type')
plt.xticks(np.arange(4), [1, 2, 3, 4], rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **3. fbs (asting blood sugar)**
# <p>(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)</p>

# In[ ]:


sizes = [len(data[data.fbs == 0]), len(data[data.fbs==1])]
labels = ['No', 'Yes']
plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# In[ ]:


sn.countplot('fbs', hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for fbs')
plt.xticks(rotation=0)
plt.xlabel('fbs > 120 mg/dl (1 = true; 0 = false)')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **3.restecg**
# (resting electrocardiographic results)

# In[ ]:


sizes = [len(data[data.restecg == 0]), len(data[data.restecg==1]), len(data[data.restecg==2])]
labels = ['Normal', 'ST-T wave abnormality', 'definite left ventricular hypertrophy by Estes criteria']
plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# In[ ]:


sn.countplot('restecg', hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for restecg')
plt.xlabel('Resting electrocardiographic measurement')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **4. exang**
# (exercise induced angina)

# In[ ]:


sizes = [len(data[data.exang == 0]), len(data[data.exang==1])]
labels = ['No', 'Yes']
plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# In[ ]:


sn.countplot('exang', hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for exang')
plt.xlabel('exercise induced angina (1=Yes: 0=No)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **5. Slope :**
# The slope of the peak exercise ST segment
# 

# In[ ]:


sizes = [len(data[data.slope == 0]), len(data[data.slope==1]), len(data[data.slope==2])]
labels = ['Upsloping', 'Flat', 'Downssloping']
plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# In[ ]:


sn.countplot('slope', hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **6. thal :**
# A blood disorder called thalassemia

# In[ ]:


sn.countplot('thal', data=data)
plt.title('Frequency for thal')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


sn.countplot('thal', hue='target', data=data, palette='mako_r')
plt.title('Heart Disease Frequency for thal')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **7. Age :**
# 

# In[ ]:


data.age.hist(bins=30)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 6))
sn.countplot('age', hue='target', data=data)
plt.title('Heart Disease Frequency for Age')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **8. Chol :**
# serum cholestoral in mg/dl
# 
# <font color='red'>Here we ca see, how Disease depends on cholestoral.

# In[ ]:


plt.hist([data.chol[data.target==0], data.chol[data.target==1]], bins=20,color=['green', 'orange'], stacked=True)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for cholestoral ')
plt.ylabel('Frequency')
plt.plot()


# **9. thalach :**
# maximum heart rate achieved
# 

# In[ ]:


plt.hist([data.thalach[data.target==0], data.thalach[data.target==1]], bins=20,color=['green', 'orange'], stacked=True)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for maximum heart rate achieved')
plt.ylabel('Frequency')
plt.plot()


# In[ ]:


plt.figure(figsize=(12, 12))
sn.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()


# # Data Preprocessing 
# Datasets contains Categorical Data so we have to create dummy variables.<br>
# 'cp', 'thal' and 'slope' are categorical variables

# In[ ]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)


# <font color=red>We use drop_first to get k-1 dummies out of k categorical levels by removing the first level.</font>

# In[ ]:


new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.head()


# We don't need cp, thal, slope columns so we will drop them

# In[ ]:


new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data.head()


# In[ ]:


# removing target columns from dataset
X = new_data.drop(['target'], axis=1)
y = new_data.target


# In[ ]:


print(X.shape)


# **Normalize the data**
# <img src="https://beyondbacktesting.files.wordpress.com/2017/07/normalization.png?w=863" />

# In[ ]:


X = (X - X.min())/(X.max()-X.min())
X.head()


# We will split our data. 80% of our data will be train data and 20% of it will be test data.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# # Train the Models

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
# checking the score at test data
lr.score(X_test, y_test)


# **Confusion Matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr.predict(X_test))
sn.heatmap(cm, annot=True)
plt.plot()


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Setting parameters for GridSearchCV
params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
lr_model = GridSearchCV(lr,param_grid=params,cv=10)


# In[ ]:


lr_model.fit(X_train,y_train)
lr_model.best_params_


# In[ ]:


lr = LogisticRegression(C=1, penalty='l2')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[ ]:


# measure the quality of predictions
from sklearn.metrics import auc, classification_report
print(classification_report(y_test, lr.predict(X_test)))


# Our model is giving good result.

# ## Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc.score(X_test, y_test)


# Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test, svc.predict(X_test))
sn.heatmap(cm, annot=True)
plt.plot()


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators':list(range(10,30)),
         'max_depth':list(range(1,7))}
rf_model = GridSearchCV(RandomForestClassifier(),param_grid=params,cv=10)
rf_model.fit(X_train,y_train)
rf_model.best_params_ 


# In[ ]:


rfc = RandomForestClassifier(max_depth= 3, n_estimators= 17, random_state=2)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)


# In[ ]:


cm = confusion_matrix(y_test, rfc.predict(X_test))
sn.heatmap(cm, annot=True)
plt.plot()


# In[ ]:


# measure the quality of predictions
print(classification_report(y_test, rfc.predict(X_test)))


# # Gradient Boosting Classifies

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc.score(X_test, y_test)


# In[ ]:


cm = confusion_matrix(y_test, gbc.predict(X_test))
sn.heatmap(cm, annot=True)
plt.plot()


# # Gaussian NB

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)


# In[ ]:


cm = confusion_matrix(y_test, nb.predict(X_test))
sn.heatmap(cm, annot=True)
plt.plot()


# In[ ]:


# measure the quality of predictions
print(classification_report(y_test, nb.predict(X_test)))


# In[ ]:


# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, nb.predict(X_test)))


# # All Algo's Score

# In[ ]:


algo = ['LogisticRegression', 'SVC', 'RandomForest', 'GaussianNB', 'GradientBoosting']
score = [al.score(X_test, y_test) for al in [lr, svc,rfc, nb, gbc]] 


# In[ ]:


plt.grid()
plt.bar(x=algo, height=score, color=['red','green','b','pink','orange'])
plt.xticks(rotation=90)
plt.ylim((0,1))
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# In[ ]:




