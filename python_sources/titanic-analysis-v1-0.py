import pandas as pd #To read the CSV file
from sklearn.naive_bayes import GaussianNB #For applying Gaussian Naive Bayes model

#Read CSV files
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

#Drop columns that will not affect analysis
train_data=train_data.drop("PassengerId",axis=1)
train_data=train_data.drop("Name",axis=1)
train_data=train_data.drop("Ticket",axis=1)
train_data=train_data.drop("Fare",axis=1)
train_data=train_data.drop("Cabin",axis=1)
train_data=train_data.drop("Embarked",axis=1)

test_data=test_data.drop("PassengerId",axis=1)
test_data=test_data.drop("Name",axis=1)
test_data=test_data.drop("Ticket",axis=1)
test_data=test_data.drop("Fare",axis=1)
test_data=test_data.drop("Cabin",axis=1)
test_data=test_data.drop("Embarked",axis=1)

#Data cleansing - Filling data with mean age available in the dataset
print(test_data.isnull().sum())
#print(round(train_data.Age.mean()))
train_data.Age=train_data.Age.fillna(round(train_data.Age.mean()))
test_data.Age=test_data.Age.isna(round(test_data.Age.mean())
#print(train_data.isnull().sum())