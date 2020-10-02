import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#encode female as 0 and male as 1, since RF expects only numeric features
train['Gender']  = train['Sex'].map( {'female':0, 'male':1} ).astype(int)
test['Gender']  = test['Sex'].map( {'female':0, 'male':1} ).astype(int)

#print(train[ (train['Age'] == max(train['Age']))])

#print(train[['Age','Name']].sort(['Age']))

'''
for i in range(0.0, 80.0, 10.0):
    print(i, train[(train['Age']>=i & train['Age']<(i+10))]['Name'])
'''

#normalize fare for sibsp and parch
#train['Fare_norm'] = train['Fare']/(train['SibSp']+train['Parch']+1)
#test['Fare_norm'] = test['Fare']/(test['SibSp']+test['Parch']+1)
train['Fare_norm'] = train['Fare']
test['Fare_norm'] = test['Fare']

train.loc[ (train['Fare']==0), 'Fare_norm' ] = np.nan
test.loc[ (test['Fare']==0), 'Fare_norm' ] = np.nan

#get median fare based on class
average_fare = np.zeros(3)
for i in range(0, 3):
    average_fare[i] = train[ (train['Pclass'] == i+1) ]['Fare_norm'].dropna().median()
    
#fill in fare
for i in range(0, 3):
    train.loc[ (train['Pclass']==i+1) & (train['Fare_norm'].isnull())\
               , 'Fare_norm' ] = average_fare[i]
    test.loc[ (test['Pclass']==i+1) & (test['Fare_norm'].isnull())\
               , 'Fare_norm' ] = average_fare[i]

# Estimate ages for passengers with null values
train['AgeFill'] = train['Age']
test['AgeFill'] = test['Age']

median_age = np.zeros((2, 3))

# Get median ages based on gender and passenger class
for i in range(0, 2):
    for j in range(0, 3):
        median_age[i, j] = train[ (train['Gender'] == i) & (train['Pclass'] == j+1) ]\
                           ['Age'].dropna().median()

# Fill in median ages where 'Age' value is null
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train['Gender']==i) & (train['Pclass']==j+1) & (train['Age'].isnull())\
               , 'AgeFill' ] = median_age[i, j]
        test.loc[ (train['Gender']==i) & (test['Pclass']==j+1) & (test['Age'].isnull())\
               , 'AgeFill' ] = median_age[i, j]
               

train['Embarked_norm']  = train['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
test['Embarked_norm']  = test['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
#fill in embarked
embarked = np.zeros(3)
for i in range(0, 3):
    embarked[i] = train[ (train['Pclass'] == i+1)]['Embarked_norm'].dropna().median()
for i in range(0, 3):
    train.loc[ (train['Pclass']==i+1) & (train['Embarked_norm'].isnull())\
               , 'Embarked_norm' ] = embarked[i]
    test.loc[ (test['Pclass']==i+1) & (test['Embarked_norm'].isnull())\
               , 'Embarked_norm' ] = embarked[i]

#extract title
train['Title'] = train['Name'].str.extract(',\s(\w+)\.')
train.loc[(train['Title'] != 'Mrs') & (train['Title'] != 'Mr') & (train['Title'] != 'Master') & (train['Title'] != 'Miss'),'Title'] = 'Rare'
train['Title'] = pd.Categorical(train['Title']).codes
test['Title'] = test['Name'].str.extract(',\s(\w+)\.')
test.loc[(test['Title'] != 'Mrs') & (test['Title'] != 'Mr') & (test['Title'] != 'Master') & (test['Title'] != 'Miss'),'Title'] = 'Rare'
test['Title'] = pd.Categorical(test['Title']).codes

#extract cabin
train['Cabin'] = train['Cabin'].str.extract('^(\w)')
train['Cabin_code'] = pd.Categorical(train['Cabin']).codes
test['Cabin'] = test['Cabin'].str.extract('^(\w)')
test['Cabin_code'] = pd.Categorical(test['Cabin']).codes

train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)

train_data = train.drop(['PassengerId'], axis=1).values
test_data  = test.drop(['PassengerId'], axis=1).values

#recognizer = RandomForestClassifier(n_estimators = 500)
recognizer = SVC(kernel = 'linear')
#recognizer = MLPClassifier(hidden_layer_sizes=(500, ), learning_rate='adaptive')
#recognizer = linear_model.Lasso(alpha=0.005)

'''
print("cross validation score")
score = cross_val_score(recognizer, x_tr, y_tr)
score = np.mean(score)
print(score)
'''
print(train.columns)
print(test.columns)

#print(train_data)

recognizer.fit(train_data[:, 1:],train_data[:, 0])
print("in-sample score")
print(recognizer.score(train_data[:, 1:],train_data[:, 0]))
print("feature importance")
#print(recognizer.feature_importances_)
print(recognizer.coef_)
print("cross validation score")
score = cross_val_score(recognizer, train_data[:, 1:],train_data[:, 0])
score = np.mean(score)
print(score)

prediction = recognizer.predict(test_data)

outdict = {'PassengerId': test['PassengerId'], 'Survived': prediction.astype(int)}
output = pd.DataFrame(data=outdict)

output.to_csv('titanic_survival_rand_forest.csv', index=False)



