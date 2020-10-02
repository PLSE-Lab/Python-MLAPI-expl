import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import tree,metrics,base
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print ("Dimension of train data {}".format(train.shape))
print ("Dimension of test data {}".format(test.shape))

print (train.Survived.value_counts())


##Data Cleaning

#print(train.isnull().sum())
#print(test.isnull().sum())


##Replacing Embarked with C because it is the most common value.

#print("After replacement of Embarked")
_ = train.set_value(train.Embarked.isnull(), 'Embarked', 'C')
#print(train.isnull().sum())




##Fare only 1 value is missing from test data. We use simple most common
    ## fare for class = 3 and embarked = S because this is the value for missing fare.
#print(test.isnull().sum())
##test[(test.Pclass==3)&(test.Embarked=='S')].Fare.value_counts().head()

_ = test.set_value(test.Fare.isnull(), 'Fare', 8.05)
#print(test.isnull().sum())


#print(train.isnull().sum())
#print(test.isnull().sum())


##Drop cabin because there is no logical correlation.
    ##P-lus, the correlation of cabin is with fare. Fare can take care of that.


#####FULL DATA#####
full = pd.concat([train, test], ignore_index=True)

#Normalizing fares
scaler = StandardScaler()
full['NorFare'] = pd.Series(scaler.fit_transform(full.Fare.values.reshape(-1,1)).reshape(-1), index=full.index)

_=train.set_value(train.index,'Fare',full[:891]['Fare'].values)
_=test.set_value(test.index,'Fare',full[891:]['Fare'].values)

print(full.isnull().sum())


##Encoding Sex
train.Sex = np.where(train.Sex=='female', 0, 1)
test.Sex = np.where(test.Sex=='female', 0, 1)
full.Sex = np.where(full.Sex=='female',0,1)
#Encoding Embarked
embarked_dict = {'S':0,'C':1,'Q':2}
train['Embarked'] = train['Embarked'].map(embarked_dict)

test['Embarked'] = test['Embarked'].map(embarked_dict)

full['Embarked'] = full['Embarked'].map(embarked_dict)

##Predict Age

X = full[~full.Age.isnull()].drop('Age', axis=1).drop('Name',axis=1).drop('Ticket',axis=1).drop('Cabin',axis=1).drop('Survived',axis=1)
#print(X.head())
#print(X.isnull().sum())
y = full[~full.Age.isnull()].Age
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)




clf = tree.DecisionTreeRegressor()
clf.fit(X, y)
predictAge = clf.predict(full[full.Age.isnull()].drop('Age', axis=1).drop('Name',axis=1).drop('Ticket',axis=1).drop('Cabin',axis=1).drop('Survived',axis=1))
full.set_value(full.Age.isnull(),'Age',predictAge)

print("After filling age")
print(full.isnull().sum())
#predictionR = clf.score(X,y)
#print(predictionR)



####Actual Model Building

X = full[~full.Survived.isnull()].drop(['Survived'], axis=1).drop(['Cabin'],axis=1).drop(['Name'],axis=1).drop(['Ticket'],axis=1).drop(['PassengerId'],axis=1)#.drop(['NorFare'],axis=1)#.drop(['Fare'],axis=1)
y = full[~full.Survived.isnull()].Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = MLPClassifier(activation = 'logistic',solver='lbfgs', alpha=0.0001,
                     random_state=1)
clf.fit(X_train,y_train)

predictedSurvival = clf.predict(X_test)
precision = metrics.precision_score(y_test, predictedSurvival, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedSurvival, normalize=True, sample_weight=None)

print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))


clf.fit(X,y)

toPredict = full[full.Survived.isnull()].drop(['Survived'], axis=1).drop(['Cabin'],axis=1).drop(['Name'],axis=1).drop(['Ticket'],axis=1).drop(['PassengerId'],axis=1)#.drop(['NorFare'],axis=1)#.drop(['Fare'],axis=1)
#toPredictCheck = full[full.Survived.isnull()].drop(['Survived'], axis=1).drop(['Cabin'],axis=1).drop(['Name'],axis=1).drop(['Ticket'],axis=1)#.drop(['PassengerId'],axis=1)#.drop(['NorFare'],axis=1)
#print(toPredictCheck.head())

output = clf.predict(toPredict)

submission = pd.DataFrame({
        "PassengerId": full[full.Survived.isnull()]["PassengerId"],
        "Survived": output
    })
submission.to_csv('output_NN.csv', index=False)
