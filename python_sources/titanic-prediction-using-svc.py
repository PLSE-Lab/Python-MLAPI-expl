import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

#reading the training data
df = pd.read_csv('/kaggle/input/titanic/train.csv')

print(df.info()) # gives info about no. of non-null values and Data type  

#Encoding the sex attribute
encoder_sex = LabelEncoder()
df['Sex'] = encoder_sex.fit_transform(df['Sex'])

#Encoding the Embarked attribute
binarizer_embarked = LabelBinarizer()
df = df.dropna(subset=['Embarked']) #dropping the rows with null value for embarked ( Just 2 rows )
x = binarizer_embarked.fit_transform(df['Embarked'])
table = []
for i in range(0,len(x)):
    row = {}
    row['S'] = x[i][0]
    row['C'] = x[i][1]
    row['Q'] = x[i][2]
    table.append(row)
binarized_embarked = pd.DataFrame(table)
df['S'] = binarized_embarked['S'].values
df['C'] = binarized_embarked['C'].values
df['Q'] = binarized_embarked['Q'].values
df.drop('Embarked', axis=1, inplace=True)

#replacing null values with mean for age attribute
df['Age'] = df['Age'].fillna(df['Age'].mean())

#scaling fare attribute with the range 0 to 1
scaler_fare = MinMaxScaler()
df['Fare'] = scaler_fare.fit_transform(df['Fare'].values.reshape(-1,1))

#getting the labels from the training data
label = df['Survived']

# dropping the attributes with very low correlation with label 
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)
print(df.corr()) # to check correlation of attributes with the label
df.drop('Survived', axis=1, inplace=True)

# checking the performance of svc model on training model using cross validation
from sklearn.svm import SVC
svc = SVC(gamma=2, C=1)
scores = cross_val_score(svc, df, label, cv=10)
print(scores.mean(),scores.std())

# training the svc model
svc.fit(df,label)

prediction_train= svc.predict(df)
# checking accuracy of svc model on training data
print(metrics.accuracy_score(label, prediction_train))

#reading the test data
test = pd.read_csv('/kaggle/input/titanic/test.csv')

#setting/cleaning up the test data
x = binarizer_embarked.transform(test['Embarked'])
table = []
for i in range(0,len(x)):
    row = {}
    row['S'] = x[i][0]
    row['C'] = x[i][1]
    row['Q'] = x[i][2]
    table.append(row)
binarized_embarked = pd.DataFrame(table)
test['S'] = binarized_embarked['S'].values
test['C'] = binarized_embarked['C'].values
test['Q'] = binarized_embarked['Q'].values
test_passengerId = test['PassengerId']
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age', 'Embarked'], axis=1, inplace=True)
test['Sex'] = encoder_sex.transform(test['Sex'])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test['Fare'] = scaler_fare.transform(test['Fare'].values.reshape(-1,1))

# making prediction on test data using the trained svc model
test_pred = svc.predict(test)

# converting the predictions in the required format
table = []
for i in range(0,len(test_pred)):
    row = {}
    row['PassengerId'] = test_passengerId[i]
    row['Survived'] = test_pred[i]
    table.append(row)
answer = pd.DataFrame(table)

#saving the predictions in a csv file
answer.to_csv("titanic_answer.csv",index=False)