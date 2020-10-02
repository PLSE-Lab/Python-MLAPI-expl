# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df_train = pd.read_csv('../input/titanic/train.csv')

# Any results you write to the current directory are saved as output.

df_train.head()

#Check out for missing data from the train dataset

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train,palette='winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df_train.drop('Cabin',axis=1,inplace=True)

df_train.dropna(inplace=True)

sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

df_train = pd.concat([df_train,sex,embark],axis=1)

df_train.head()

#Now since the data is ready to be applied to Machine Learning algortihms, lets import the test data as well

df_test = pd.read_csv('../input/titanic/test.csv')

df_test.head()

#The test data also has discrepanicies that need to be addressed by imputing age values

df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)

#We use the same function being used for Train Data on the Test Data as well

sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df_test.drop('Cabin',axis=1,inplace=True)

df_test.dropna(inplace=True)

sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)

df_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

df_test = pd.concat([df_test,sex,embark],axis=1)

df_test.head()


#Let us now apply the following machine learning algorithms and check which performs the best 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test
X_train.shape, Y_train.shape, X_test.shape

#Logisitc Regression

logreg_titanic = LogisticRegression()
logreg_titanic.fit(X_train, Y_train)
Y_prediction = logreg_titanic.predict(X_test)
acc_log_titanic = round(logreg_titanic.score(X_train, Y_train) * 100, 2)
acc_log_titanic

#Random Forest Classiifier

random_forest = RandomForestClassifier(n_estimators=205)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

#KNN Classifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

#Compare all the above scores to find the best fit

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest Classifier', 
              'Linear SVC', 'Gaussian NB', 'KNN'],
    'Score': [acc_log_titanic, acc_random_forest, acc_linear_svc, 
              acc_gaussian, acc_knn]})
models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)