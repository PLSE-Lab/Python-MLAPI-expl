# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
scores={}

def LinearRegressionModel(X_train,y_train,X_test):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    #Predict results for Test set
    y_pred = classifier.predict(X_test)
    scores['LogisticRegression'] = classifier.score(X_train, y_train)    
    return y_pred

def KenelSVMModel(X_train,y_train,X_test):
    classifier = SVC(kernel='rbf', random_state=0,C=1.0,gamma=0.1)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    scores['KernelSVM'] = classifier.score(X_train, y_train)
    return y_pred

def RFModel(X_train,y_train,X_test):
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train,y_train)    
    y_pred = classifier.predict(X_test)
    scores['RandomForest'] = classifier.score(X_train, y_train)
    return y_pred

def NaiveBayesModel(X_train,y_train,X_test):
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)    
    y_pred = classifier.predict(X_test)
    scores['NaiveBayes'] = classifier.score(X_train, y_train)
    return y_pred

def KNNModel(X_train,y_train,X_test):
    classifier=KNeighborsClassifier(n_neighbors=65,metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    scores['KNN'] = classifier.score(X_train, y_train)
    return y_pred


# Importing train dataset.
dataset_original_train = pd.read_csv('../input/train.csv',header=0)
#Removing few variables from dataset as they won't be much helpful in prediction
colNamesToDrop=['PassengerId','Name','Ticket','Cabin']
dataset_train = dataset_original_train.drop(colNamesToDrop,axis=1)
#Filling in NaN data with mean and most repeated values
dataset_train["Embarked"] = dataset_train["Embarked"].fillna("S")
dataset_train["Age"].fillna(dataset_train["Age"].median(), inplace=True)

# Importing test dataset
dataset_original_test = pd.read_csv('../input/test.csv',header=0)
dataset_test = dataset_original_test.drop(colNamesToDrop,axis=1)
dataset_test["Age"].fillna(dataset_test["Age"].median(), inplace=True)
dataset_test["Fare"].fillna(dataset_test["Fare"].median(), inplace=True)
dataset_test["Embarked"] = dataset_test["Embarked"].fillna("S")

#Convert DataFrame to Array
X_train = dataset_train.iloc[:,1:8].values
#get dependent variable from the train set
y_train = dataset_train.iloc[:, 0].values
X_test = dataset_test.iloc[:].values


# Encoding categorical data of both train and test dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, (1)] = labelencoder_X.fit_transform(X_train[:, (1)])
X_train[:, (6)] = labelencoder_X.fit_transform(X_train[:, (6)])
X_test[:, (1)] = labelencoder_X.fit_transform(X_test[:, (1)])
X_test[:, (6)] = labelencoder_X.fit_transform(X_test[:, (6)])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

#Avoid Dummy variable trap
X_train=X_train[:,1:]
X_test=X_test[:,1:]

#Peforming Feature Scaling on both test and train datasets
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#The sweet part : Evaluate models
Linear_ypred = LinearRegressionModel(X_train,y_train,X_test)
kSVM_ypred = KenelSVMModel(X_train,y_train,X_test)
RF_ypred = RFModel(X_train,y_train,X_test)
NB_ypred = NaiveBayesModel(X_train,y_train,X_test)
KNN_ypred = KNNModel(X_train,y_train,X_test)

df=pd.DataFrame([[key,value] for key,value in scores.items()],columns=["Algorithm","Score"])

sns.factorplot(x="Algorithm", y="Score", hue="Algorithm", data=df, kind="bar");

#Even though i see RF model is having 97% of score- I feel that, it is overfitting. 
#Hence i would like to consider SVM model.
result = pd.DataFrame({
        "PassengerId": dataset_original_test["PassengerId"],
        "Survived": kSVM_ypred
    })
result.to_csv('submission.csv', index=False)

# Titanic example feels like a verfy good kickstart for ML model evaluation.
#Comments are welcome.
#Thank you