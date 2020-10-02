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

# Loading the data to read

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.info()
test_data.info()

train_data.describe()
test_data.describe()

train_data.isnull().sum()
test_data.isnull().sum()

# feature selection..

train_data['Survived'].value_counts(normalize=True)

train_data.groupby('Pclass').Survived.mean()

train_data.groupby(['Pclass', 'Sex']).Survived.mean()

train_data.corr()
test_data.corr()

data = [train_data,test_data]
for dataset in data:
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    dataset['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
    dataset['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
    dataset['Fam_size'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset["Embarked"].fillna("S", inplace = True)
    dataset['Age'].fillna(round(np.mean(train_data['Age'])), inplace = True)
    dataset['Fare'].fillna(round(np.mean(train_data['Fare'])), inplace = True)
    
    
y = train_data['Survived']
features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked','Age','Fam_size']
X = train_data[features] 
test_data = test_data[features]

genders = {"male": 0, "female": 1}
ports = {"S": 0, "C": 1, "Q": 2}
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev":6}
data = [X, test_data]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
    dataset['Embarked'] = dataset['Embarked'].map(ports)    
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].map(titles) 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data = scaler.fit_transform(test_data)
print(X.shape)
print(test_data.shape)
print(y.shape)


# Applying various Algorithms 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,stratify = y)

# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
y_pred_SGDC = sgd.predict(X_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=500)
random_forest.fit(X_train, y_train)
y_pred_RandomForest = random_forest.predict(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_LogisticRegression = logreg.predict(X_test)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)
y_pred_GNB = gaussian.predict(X_test) 

# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron(max_iter=10)
perceptron.fit(X_train, y_train)
y_pred_Perceptron = perceptron.predict(X_test)

# Linear Support Vector Machine
from sklearn.svm import SVC, LinearSVC
linear_svc = LinearSVC(max_iter=30)
linear_svc.fit(X_train, y_train)
y_pred_SVC = linear_svc.predict(X_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
y_pred_DTC = decision_tree.predict(X_test)  

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 12) 
knn.fit(X_train, y_train)  
y_pred_KNN = knn.predict(X_test)

# Checking the accuracy and other parameters from applied algorithms

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
score = [y_pred_SGDC, y_pred_RandomForest,y_pred_LogisticRegression, y_pred_GNB,y_pred_Perceptron, y_pred_SVC, y_pred_DTC, y_pred_KNN ]
titles =['SGDClassification','Random Forest','Logistic Regression','Naive Bayes', 'Preceptron','Linear SVC','Decision Tree','KNN']


 
    
for dataset,i in zip(score, range(7)):
    print("\nResults for", titles[i],":\n")
    result = confusion_matrix(y_test, dataset)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, dataset)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test,dataset)
    print("Accuracy:",result2)

    
# From the above results Random forest gives high accuracy

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=500)
random_forest.fit(X_train, y_train)
y_pred_RandomForest = random_forest.predict(test_data)

test = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred_RandomForest})
output.to_csv('my_submission.csv', index=False)

