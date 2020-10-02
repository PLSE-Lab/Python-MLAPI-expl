# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import  cross_validation, preprocessing #data sampling,model and preprocessing 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
titanic_df=pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.


titanic_df.head() # print few data


""" Data exploration and processing"""

titanic_df['Survived'].mean()


titanic_df.groupby('Pclass').mean()


class_sex_grouping = titanic_df.groupby(['Pclass','Sex']).mean()

class_sex_grouping


class_sex_grouping['Survived'].plot.bar()       



group_by_age = pd.cut(titanic_df["Age"], np.arange(0, 90, 10))

age_grouping = titanic_df.groupby(group_by_age).mean()

age_grouping['Survived'].plot.bar()     


titanic_df.count()


titanic_df = titanic_df.drop(['Cabin'], axis=1)   


titanic_df = titanic_df.dropna()    


titanic_df.count()



""" Data preprocessing function"""

def preprocess_titanic_df(df):

    processed_df = df.copy()

    le = preprocessing.LabelEncoder()

    processed_df.Sex = le.fit_transform(processed_df.Sex)

    processed_df.Embarked = le.fit_transform(processed_df.Embarked)

    processed_df = processed_df.drop(['PassengerId','Name','Ticket'],axis=1)

    return processed_df

    
processed_df = preprocess_titanic_df(titanic_df)



X = processed_df.drop(['Survived'], axis=1) # Features dataset

y = processed_df['Survived'] # Target variable


#train test split

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=50)


#X_train = pd.DataFrame(X_train)

#Model Implementation

from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(criterion = "entropy",max_depth=5,min_samples_split = 10 ) # Define model

clf_dt.fit (X_train, y_train) # Fit model with your data

predictions = clf_dt.predict(X_test) # Preditions on test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)


#Decision Tree

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier


model =  RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy_score(y_test,pred)

