import numpy as np
import pandas as pd
import csv as csv 
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
total = pd.concat([train, test]) #The entire set

#Now let's fill the missing value of Fare
mean_fare = total[["Pclass", "Fare"]].groupby(['Pclass'],as_index=False).mean()
for i in range(0,3):
    total.loc[ total['Fare'].isnull() & total['Pclass'] == i+1, 'Fare' ] = mean_fare['Fare'][i]

#Now let's fill the missing value of Embarked
embarked_matrix = total[["Embarked", "PassengerId"]].groupby(["Embarked"],as_index=False).count()
embarked_matrix_max = embarked_matrix.max()
total.loc[ total["Embarked"].isnull(), "Embarked"] = embarked_matrix_max["Embarked"]

#Now let's extract the title and family size and  club it and assign numbers to it and other variables depending on probability of survival
total['Title'] = total['Name'].str.replace('(.*, )|(\\..*)', '')
total['Title'] = total['Title'].replace(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'special_title')
total['Title'] = total['Title'].replace(['Mlle', 'Ms'], 'Miss')
total['Title'] = total['Title'].replace('Mme', 'Mrs')
total['Title_num'] = total['Title'].map( {'Mrs':0, 'Miss':1, 'Master':2, 'special_title':3, 'Mr':4} ).astype(int) 
total['FamilySize'] = total['SibSp'] + total['Parch'] + 1
total['FamilySizeCategories'] = total['FamilySize']
total.loc[ total['FamilySizeCategories'] == 1, 'FamilySizeCategories' ] = 0 #Singleton
total.loc[ (total['FamilySizeCategories'] > 1) & (total['FamilySizeCategories'] < 5) , 'FamilySizeCategories' ] = 1 #Small
total.loc[ total['FamilySizeCategories'] > 4, 'FamilySizeCategories' ] = 2 #Large
total['FamilySizeCategories_num'] = total['FamilySizeCategories'].map( {0:1, 1:0, 2:2} ).astype(int)
total['Embarked_num'] = total['Embarked'].map( {'C':0, 'Q':1, 'S':2} ).astype(int)
total['Pclass_num'] = total['Pclass'] - 1
total['Sex_num'] = total['Sex'].map( {'female':0, 'male':1} ).astype(int)
total['Fare_num'] = pd.qcut(total['Fare'], 4, labels=[3, 2, 1, 0])

#Now let's fill the missing value of Age
mean_age_titlenum_pclass = total[["Title_num", "Pclass", "Age"]].groupby(["Title_num", "Pclass"]).mean()
for i in range(0,5):
    for j in range(1,4):
        if (i!=3) | (j!=3):
            total.loc[ (total['Age'].isnull()) & (total['Title_num'] == i) & (total['Pclass'] == j), 'Age' ] = mean_age_titlenum_pclass['Age'][i][j]

#Now let's drop irrelevant columns
total = total.drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Ticket', 'FamilySize', 'Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Title', 'FamilySizeCategories'], axis=1)

#print( total[["Embarked_num", "Survived"]].groupby(['Embarked_num'],as_index=False).mean() )
#print( total[["Pclass_num", "Survived"]].groupby(['Pclass_num'],as_index=False).mean() )
#print( total[["Sex_num", "Survived"]].groupby(['Sex_num'],as_index=False).mean() )
#print( total[["Title_num", "Survived"]].groupby(['Title_num'],as_index=False).mean() )
#print( total[["FamilySizeCategories_num", "Survived"]].groupby(['FamilySizeCategories_num'],as_index=False).mean() )
#print( total[["Fare_num", "Survived"]].groupby(['Fare_num'],as_index=False).mean() )

print(total.info())

train = total[0:890]
test = total[891:1309]
X_train = train.drop(['Survived', 'PassengerId'],axis=1)
Y_train = train["Survived"]
X_test  = test.drop(['Survived', 'PassengerId'],axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(logreg.score(X_train, Y_train))

output = Y_pred.astype(int)
ids = test['PassengerId'].values
predictions_file = open("titanic_predict.csv", "w") # Python 3
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

