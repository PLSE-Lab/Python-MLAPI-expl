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
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:41:09 2019

@author: Abhijit
"""
import os
print(os.listdir("../input"))

# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing datasets
#train=pd.read_csv(r"C:\Users\USER\Desktop\Python\My_Project\Titanic\train.csv")
#test = pd.read_csv(r"C:\Users\USER\Desktop\Python\My_Project\Titanic\test.csv")

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

Y_train= train.iloc[:, 1].values
train.drop(['Survived'], inplace = True, axis= 1)
dataset= pd.concat([train, test], axis=0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataset.info()
dataset.head()
# Checking the presence of Null values
dataset.isnull().sum()

#Age             263
#Fare              1
#Cabin          1014
#Embarked          2

# Age             263
dataset.groupby(['Pclass']).mean()
#       Age
#PC1 39.159930
#PC2 29.506705
#PC3 24.816367

dataset.groupby(['Sex']).mean()
#       Age
#Female 28.687088
#Male   30.58522

dataset.groupby(['Embarked']).mean()
#   Age
#C 32.332170
#Q 28.630000
#S 29.245205
    
dataset['Age'][(dataset['Sex'] == 'female') & (dataset['Pclass'] == 1)].mean() #37

dataset['Age'][(dataset['Sex'] == 'female') & (dataset['Pclass'] == 2)].mean() #27

dataset['Age'][(dataset['Sex'] == 'female') & (dataset['Pclass'] == 3)].mean() #22

dataset['Age'][(dataset['Sex'] == 'male') & (dataset['Pclass'] == 3)].mean() #26

dataset['Age'][(dataset['Sex'] == 'male') & (dataset['Pclass'] == 2)].mean() #31

dataset['Age'][(dataset['Sex'] == 'male') & (dataset['Pclass'] == 1)].mean() #41
        
""" Writing a Function """
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):

        if Pclass == 1 and Sex == 'female':
            return 37

        elif Pclass == 2 and Sex == 'female':
            return 27
        if Pclass == 3 and Sex == 'female':
            return 22

        elif Pclass == 1 and Sex == 'male':
            return 41
        if Pclass == 2 and Sex == 'male':
            return 31

        elif Pclass == 3 and Sex == 'male':
            return 26

        else:
            return 24

    else:
        return Age
# Applyting the function

# Filling NAN values of Age
dataset['Age'] = dataset[['Age','Pclass', 'Sex']].apply(impute_age,axis=1)

dataset.isnull().sum().sort_values(ascending = False)
dataset[dataset.Fare.isnull() == True]

#Filling NAN values of Fare
dataset['Fare'][(dataset['Pclass']==3) & (dataset['Sex']=='male')].mean() # 12
dataset['Fare'].fillna(12, inplace = True)

dataset[(dataset['Embarked'].isnull()== True)]

# Filling NAN values of Embarked
dataset['Embarked'][(dataset['Pclass']==1) & (dataset['Sex']=='female')].value_counts()
#C    71
#S    69
#Q     2
dataset['Embarked'].fillna('C', inplace = True)

dataset.isnull().sum()

# dropping CABIN

dataset.drop('Cabin', axis = 1, inplace = True)
dataset.head()

dataset.drop('Ticket', axis = 1, inplace = True)
dataset.head()

# Now treating the categorical values
Sex= pd.get_dummies(dataset.Sex, drop_first = True)
Embarked= pd.get_dummies(dataset.Embarked, drop_first = True)

new_dataset= pd.concat([Sex, Embarked, dataset], axis =1)
new_dataset.head()

new_dataset.drop(['Name','Sex', 'Embarked'], axis = 1, inplace = True)

new_dataset.head()
new_dataset.drop(['PassengerId'], axis = 1, inplace = True)

#now we need to split the dataset
Y_train # We already have
X_train= new_dataset.iloc[0:891, :].values
X_test = new_dataset.iloc[891:, :].values

# now try logistic regression
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression()
LR_classifier.fit(X_train, Y_train)
 
Y_pred_LR = LR_classifier.predict(X_test)



# Applying decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier()
DT_classifier = DT_classifier.fit(X_train, Y_train)

Y_pred_DT = DT_classifier.predict(X_test) 

# Applying Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators= 300)
RF_classifier.fit(X_train, Y_train)

Y_pred_RF = RF_classifier.predict(X_test)
# Applying NAYEV BAYES CLASSIFIER
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# now applying k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier= KNeighborsClassifier(n_neighbors = 5, metric='minkowski')
KNN_classifier.fit(X_train, Y_train)

Y_pred_KNN = KNN_classifier.predict(X_test)

# now applying NaiveBayes Classifier
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, Y_train)

Y_pred_NB = NB_classifier.predict(X_test)

# now trying it with svm
from sklearn.svm import SVC
SVC_classifier = SVC()
SVC_classifier.fit(X_train, Y_train)

Y_pred_SVC = SVC_classifier.predict(X_test)