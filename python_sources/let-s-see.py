#!/usr/bin/env python
# coding: utf-8

# 

# 

# 

# 

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning
from sklearn import preprocessing
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# EDA
data = pd.read_csv("../input/armenian_pubs.csv" )
data.head()


# In[ ]:


data.info()
print('=============================\n', data.notnull().sum())


# In[ ]:


data.shape


# In[ ]:


columns = data.columns
columns
# Notice that there are 'Age ', 'Gender ', 'Income ', not 'Age', 'Gender', 'Income'.
# In other words, we'd better remove the additional space.
data.columns = ['Timestamp', 'Age', 'Gender', 'Income', 'Occupation', 'Fav_Pub', 'WTS', 'Freq', 'Prim_Imp', 'Sec_Imp', 'Stratum', 'Lifestyle', 'Occasions']
columns = data.columns
columns


# In[ ]:


# Age
sns.countplot('Age',data=data)
# Or we can use matplotlib
'''
Agedata = data['Age']
Agebins = list(range(Agedata.min(), Agedata.max()))
plt.hist(Agedata, bins=Agebins)
plt.title('Age')
plt.show()
'''


# In[ ]:


# Gender
sns.countplot('Gender',data=data)
plt.show()


# In[ ]:


# Income
Incomedata = data['Income']
print(Incomedata.notnull().sum())
print(Incomedata.describe())
sns.boxplot(Incomedata)


# In[ ]:


# Notice that there are some outliers, let's remove them and plot them again for the detail
Incomedata_cleaned = Incomedata[Incomedata < 1000000]
sns.boxplot(Incomedata_cleaned)


# In[ ]:


# The rest...yes, still countplot...I love it :)
cols = ['Occupation', 'Freq', 'Prim_Imp', 'Sec_Imp', 'Stratum', 'Lifestyle', 'Occasions']
fig = plt.figure(figsize=(12,30))
for i in range(len(cols)):
    fig.add_subplot(len(cols),1, i+1)
    sns.countplot(cols[i], data=data)
    


# In[ ]:


# Let's try do some predict by training the data
# Try to transform str value to numeric value
# We'll use the method as fellow to do this job
le = preprocessing.LabelEncoder()
le.fit(data['Occupation'].unique())
print(le.classes_)
le.transform(data['Occupation'])


# In[ ]:


numcols = ['Age', 'Income', 'WTS']
strcols = ['Gender', 'Occupation', 'Prim_Imp', 'Sec_Imp', 'Stratum', 'Lifestyle', 'Occasions','Freq']


# Classify the age
Cdata = data.copy()

Simplage = Cdata['Age']
fig.add_subplot(311)
sns.boxplot('Age', data=Cdata)
Simplage[Simplage < 18] = 0
Simplage[(Simplage >= 18) & (Simplage <23)] = 1
Simplage[Simplage >= 23] = 2
print(Simplage.head())

fig = plt.figure(figsize=(12,24))
# Income
Incomedata = Cdata['Income']
# Many people are student with no income, so we use o to fill the missing value
Simplincome = Incomedata.fillna(0.0)
#print('==============\n', Simplincome.notnull().sum())
print(Simplincome.describe())
fig.add_subplot(312)
sns.boxplot(Simplincome)
Simplincome[Simplincome <= 2000] = 0
Simplincome[(2000 < Simplincome) & (Simplincome <= 4000)] = 1
Simplincome[(4000 < Simplincome) & (Simplincome <= 6000)] = 2
Simplincome[Simplincome > 8000] = 3
print(Simplincome.head())


# Classify the WTS
Simplwts = Cdata['WTS']
#print(Simplwts.notnull().sum())
# fill these missing values
fig.add_subplot(313)
sns.boxplot('WTS', data=Cdata)

Simplwts = Simplwts.fillna(5000)
#print(Simplwts.notnull().sum())
#print(Simplwts.describe())
# Classify
Simplwts[Simplwts <= 2000] = 0
Simplwts[(2000 < Simplwts) & (Simplwts <= 4000)] = 1
Simplwts[(4000 < Simplwts) & (Simplwts <= 6000)] = 2
Simplwts[Simplwts > 8000] = 3
#print(Simplwts.head())

newdata = pd.concat([Simplage, Simplincome, Simplwts], axis=1)
print(newdata.head())
print(newdata.notnull().sum())


# In[ ]:


# fillna and label them
ontodict = {}
def gettrans(colname):
    coldata = Cdata[colname]
    coldata = coldata.fillna(coldata.mode()[0])  # the [0] looks like indispensable
    le = preprocessing.LabelEncoder()
    le.fit(coldata.unique())
    #print(colname, '-->', le.classes_)
    ontodict[colname] = le.classes_
    newcoldata = le.transform(coldata)
    newdata[colname] = newcoldata


# In[ ]:


# str cols
print(Cdata.notnull().sum())

# fillna and label them
ontodict = {}
def gettrans(colname):
    coldata = Cdata[colname]
    coldata = coldata.fillna(coldata.mode()[0])  # the [0] looks like indispensable
    le = preprocessing.LabelEncoder()
    le.fit(coldata.unique())
    #print(colname, '-->', le.classes_)
    ontodict[colname] = le.classes_
    newcoldata = le.transform(coldata)
    newdata[colname] = newcoldata


for colname in strcols:
    gettrans(colname)

print(newdata.head())
print(newdata.notnull().sum())
print(ontodict)


# In[ ]:


newdata.head()


# In[ ]:


# Let's make Freq as the value that can be predicted by other values
x = newdata[['Age', 'Income', 'WTS', 'Gender', 'Occupation', 'Prim_Imp', 'Sec_Imp',
       'Stratum','Lifestyle', 'Occasions']]
y = newdata['Freq']

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


# Let's make Freq as the value that can be predicted by other values
x = newdata[['Age', 'Income', 'WTS', 'Gender', 'Occupation', 'Prim_Imp', 'Sec_Imp',
       'Stratum','Lifestyle', 'Occasions']]
y = newdata['Freq']

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


# predict

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
logreg.score(x_train, y_train)


# In[ ]:


# knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_test)
knn.score(x_train, y_train)


# In[ ]:



# Random Forests
random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)


# In[ ]:



