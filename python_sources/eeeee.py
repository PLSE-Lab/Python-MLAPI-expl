from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = read_csv("../input/train.csv")

data.Age[data.Age.isnull()] = data.Age.mean()
data.Fare[data.Fare.isnull()] = data.Fare.median()
MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(data.PassengerId)
data = data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex)
label.fit(data.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
data.Embarked = label.transform(data.Embarked)

test = read_csv("../input/test.csv")

test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median()
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(test.PassengerId)
test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)
label.fit(dicts['Embarked'])
test.Embarked = label.transform(test.Embarked)
target = data.Survived
train = data.drop(['Survived'], axis=1)
model_rfc = RandomForestClassifier(n_estimators = 70) 
model_rfc.fit(train, target)
result.insert(1,'Survived', model_rfc.predict(test))
result.to_csv('MYprediction.csv', index=False)