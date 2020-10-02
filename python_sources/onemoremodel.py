import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression

# This creates a pandas dataframe and assigns it to the train and test variables
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = [train, test]

# Creating new feature as Title
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


for c in combine:
	c['Title'] = c['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	c['Title'] = c['Title'].replace('Mlle', 'Miss')
	c['Title'] = c['Title'].replace('Ms', 'Miss')
	c['Title'] = c['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for c in combine:
    c['Title'] = c['Title'].map(title_mapping)
    c['Title'] = c['Title'].fillna(0)
    
train["Age"]=train["Age"].fillna(train["Age"].median())

guess_ages = np.zeros((5,1))
        
for c in combine:
    for i in range(0, 5):
        guess_df = c[(c['Title'] == i+1) ]['Age'].dropna()
        age_guess = guess_df.median()
        # Convert random age float to nearest .5 age
        guess_ages[i,0] = int( age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0,5):
        c.loc[ (c.Age.isnull()) & (c.Title == i) ,"Age"] = guess_ages[i,0]

# kid survival rate is more so replacing sex of less than 16 years passangers as kids

sex_mapping = {"male": 0, "female": 1, "kid": 2}

for c in combine:
    c.loc[c["Age"]<=16,"Sex"] = "kid"
    c['Sex'] = c['Sex'].map(sex_mapping)
    c["Fare"]=c["Fare"].fillna(train["Fare"].median())

for c in combine:
    c.loc[ c['Fare'] <= 10, 'FareB'] = 0
    c.loc[(c['Fare'] > 10) & (c['Fare'] <= 25), 'FareB'] = 1
    c.loc[(c['Fare'] > 25) & (c['Fare'] <= 55), 'FareB'] = 2
    c.loc[ c['Fare'] > 55, 'FareB'] = 3


#Creating variable for regression
X_train = train[["Sex","FareB"]].copy()
Y_train = train["Survived"]
X_test  = test[["Sex","FareB"]].copy()

'''
#Implementing logstic regression
logreg = LogisticRegression()
result = logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
predicted_probs = logreg.predict_proba(X_test)
'''

'''
#Implementing linear regression
# Create linear regression object
logreg = LinearRegression()

# Train the model using the training sets
result= logreg.fit(X_train, Y_train)
Pred_train = logreg.predict(X_train)
Y_pred = logreg.predict(X_test)

Y_pred = np.where(Y_pred<=0.5, 0,1)
'''
'''
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

print(random_forest.score(X_train, Y_train))
'''

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)


#submitting assignment
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)