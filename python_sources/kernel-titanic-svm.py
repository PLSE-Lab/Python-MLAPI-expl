# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#%matplotlib inline

# training data   # this is the path to the Iowa data that you will use
train = pd.read_csv('../input/train.csv')
# that's a classification problem
test = pd.read_csv('../input/test.csv') # many advantages no need to reduce the test data again

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.columns)
categorical_features = train.select_dtypes(include=[np.object])
print(categorical_features.columns)
train.head()

train_len = len(train)
print('length of training', train_len)
data=pd.concat(objs=[train, test], axis=0,sort=False).reset_index(drop=True)
print('length of all data', len(data))
print(pd.isnull(data).sum())

#Embarked 
data_ms_Embarked=data[data['Embarked'].isnull()]
data.groupby('Embarked').describe()

def Embarked_approx(cols):
    Embarked = cols[0]
    Fare = cols[1]
    if pd.isnull(Embarked):
        if Fare >=50:
            return 'C'
        elif Fare <=15:
            return 'Q'
        else:
            return 'S'
    else:
        return Embarked

data['Embarked'] = data[['Embarked', 'Fare']].apply(Embarked_approx, axis=1)
codes = {'C':3,'Q':2,'S':1}
data['Embarkedn'] = data['Embarked'].map(codes).astype(int)

data['Fare'].fillna(13, inplace=True) 
#data_ms=data_t[data_t['Fare'].isnull()]
#data_ms
print(pd.isnull(data).sum())

#sex
codes = {'male':0,'female':1}
data['Sexn'] = data['Sex'].map(codes).astype(int)


#title
def process_name(name):
    if name.find('Mrs')>=0:
        return 'Mrs'
    elif name.find('Miss')>=0:
        return 'Miss'
    elif name.find('Mr')>=0:
        return 'Mr'
    elif name.find('Master')>=0:
        return 'Mas'
    else: 
        #print name
        return 'else'


data['Title'] = data.Name.apply(process_name) 
codes = {'Miss':1,'Mr':2,'Mrs':3,'Mas':4,'else':5}
data['Titlen'] = data['Title'].map(codes).astype(int)
data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

data['FBand'] = pd.cut(data['FamilySize'], 5)
data[['FBand', 'Survived']].groupby(['FBand'], as_index=False).mean().sort_values(by='FBand', ascending=True)

F_labels = ['1', '2', '3', '4', '5']
data['Fgroup'] = pd.cut(data.FamilySize, range(1, 12, 2), right=True,labels=F_labels)
data.Fgroup.fillna('1',inplace=True)
data[['Fgroup', 'Survived']].groupby(['Fgroup'], as_index=False).mean().sort_values(by='Fgroup', ascending=True)

#Survived 
import seaborn as sb

data["Fare"] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sb.distplot(data["Fare"], color="m", label="Skewness : %.2f"%(data["Fare"].skew()))
g = g.legend(loc="best")
#Pclass Fare correlate 
#g = sb.distplot(data["Fare"], color="m", label="Skewness : %.2f"%(data["Fare"].skew()))
#g = g.legend(loc="best")
data["Fare"]=data["Fare"]/ data["Fare"].max()

mean_ages = data.groupby(['Sex','Pclass'])['Age'].mean()
mean_ages

def process_age(cols):
    if pd.isnull(cols['Age']):
        return mean_ages[cols['Sex'],cols['Pclass']]
    else:
        return cols['Age']
        
data['Age'] = data.apply(process_age, axis=1)

data['AgeBand'] = pd.cut(data['Age'], 5)
data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
age_labels = ['1', '2', '3', '4', '5']
data['Age_group'] = pd.cut(data.Age, range(0, 81, 16), right=True, labels=age_labels)
data[['Age_group', 'Survived']].groupby(['Age_group'], as_index=False).mean().sort_values(by='Age_group', ascending=True)

train = data[:train_len]
test = data[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]

print(len(Y_train))
#train
Predictors=['Fare','Sexn','Embarkedn','Fgroup','Titlen','Age_group','Pclass']
X_train=train[Predictors]

X_test=test[Predictors]
X_test.shape
X_train.head()

#SVM method 
from sklearn.model_selection import train_test_split
from sklearn import svm
svm = svm.SVC(C=5.0)
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)
acc_svc = round(svm.score(X_train, Y_train) * 100, 2)
print(acc_svc)

submission=pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': Y_pred
    })

#write to res.csv
import csv
submission.to_csv('res.csv', encoding='utf-8', index=False)
