





import numpy as np
import pandas as pd
from sklearn import cross_validation, tree, linear_model, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time

# Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, header=0)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, header=0)

# Pclass - imputacao dos valores ausentes
mode_pclass = train['Pclass'].dropna().mode().values

train['Pclass'].fillna(mode_pclass[0], inplace=True)
test['Pclass'].fillna(mode_pclass[0], inplace=True)

# Name - imputacao dos valores ausentes

train['Name'].fillna('x, others.', inplace=True)
test['Name'].fillna('x, others.', inplace=True)

# Name - convert


def title_type(t):
    if t == 'Mr':
        return 0
    elif t == 'Capt':
        return 1
    elif t == 'Rev':
        return 2
    elif t == 'Master':
        return 3
    elif t == 'Mrs':
        return 4
    elif t == 'Mlle':
        return 5
    elif t == 'Miss':
        return 6
    elif t == 'Dr':
        return 7
    else:
        return 8

# def title_type(t):
#     if t != 'Mr' and t != 'Capt' and t != 'Rev' and t != 'Master' and t != 'Mrs' and t != 'Mlle' and t != 'Miss' and t != 'Dr':
#         return 'Other'

title = []
for name in train['Name']:
    t1 = name.split(', ')
    t2 = t1[1].split('.')
    title.append(t2[0])

title_int = []
for name in title:
    title_int.append(title_type(name))

train['Title'] = title_int

title = []
for name in test['Name']:
    t1 = name.split(', ')
    t2 = t1[1].split('.')
    title.append(t2[0])

title_int = []
for name in title:
    title_int.append(title_type(name))

test['Title'] = title_int




# Sex - imputacao dos valores ausentes
mode_sex = train['Sex'].dropna().mode().values

sex_nan = train['Sex'][train['Sex'].isnull()]
for x in sex_nan:
    if train['Title'][x[0]] in [0, 1, 3, 2]:
        train.loc[x[0], 'Sex'] = 'male'
    elif train['Title'][x[0]] in [4, 5, 6]:
        train.loc[x[0], 'Sex'] = 'female'
    else:
        train.loc[x[0], 'Sex'] = mode_sex

sex_nan = test['Sex'][test['Sex'].isnull()]
for x in sex_nan:
    if test['Title'][x[0]] in [0, 1, 3, 2]:
        test.loc[x[0], 'Sex'] = 'male'
    elif test['Title'][x[0]] in [4, 5, 6]:
        test.loc[x[0], 'Sex'] = 'female'
    else:
        test.loc[x[0], 'Sex'] = mode_sex


# Age - imputacao dos valores ausentes
mode_age = train['Age'].dropna().mode().values

train['Age'].fillna(mode_age[0], inplace=True)
test['Age'].fillna(mode_age[0], inplace=True)

ageGroup = []
for i in range(len(train['Age'])):
    if train['Age'][i] > 0 and train['Age'][i] <= 10:
        ageGroup.append('child')
    elif train['Age'][i] > 10 and train['Age'][i] <= 18:
        ageGroup.append('adolescent')
    elif train['Age'][i] > 18 and train['Age'][i] <= 50:
        ageGroup.append('adult')
    elif train['Age'][i] > 50 and train['Age'][i] <= 100:
        ageGroup.append('old')
    else:
        ageGroup.append('unknown')

train['AgeGroup'] = ageGroup

ageGroup = []
for i in range(len(test['Age'])):
    if test['Age'][i] > 0 and test['Age'][i] <= 10:
        ageGroup.append('child')
    elif test['Age'][i] > 10 and test['Age'][i] <= 18:
        ageGroup.append('adolescent')
    elif test['Age'][i] > 18 and test['Age'][i] <= 50:
        ageGroup.append('adult')
    elif test['Age'][i] > 50 and test['Age'][i] <= 100:
        ageGroup.append('old')
    else:
        ageGroup.append('unknown')

test['AgeGroup'] = ageGroup








# SibSp - imputacao dos valores ausentes
mode_sibSp = train['SibSp'].dropna().mode().values

train['SibSp'].fillna(mode_sibSp[0], inplace=True)
test['SibSp'].fillna(mode_sibSp[0], inplace=True)



# Parch - imputacao dos valores ausentes
mode_parch = train['Parch'].dropna().mode().values

train['Parch'].fillna(mode_parch[0], inplace=True)
test['Parch'].fillna(mode_parch[0], inplace=True)


#combinacao entre sibSp e parch

train['numFam'] = train['SibSp'] + train['Parch']
test['numFam'] = test['SibSp'] + test['Parch']


for i in range(len(train['numFam'])):
    if train['numFam'][i] > 7:
        train.loc[i, 'numFam'] = 'grande'
    elif train['numFam'][i] <= 7 and train['numFam'][i] >= 4:
        train.loc[i, 'numFam'] = 'media'
    elif train['numFam'][i] < 4 and train['numFam'][i] > 0:
        train.loc[i, 'numFam'] = 'pequena'
    else:
        train.loc[i, 'numFam'] = 'sem'

train = train.drop(['SibSp', 'Parch'], axis=1)

for i in range(len(test['numFam'])):
    if test['numFam'][i] > 7:
        test.loc[i, 'numFam'] = 'grande'
    elif test['numFam'][i] <= 7 and test['numFam'][i] > 3:
        test.loc[i, 'numFam'] = 'media'
    elif test['numFam'][i] <= 3 and test['numFam'][i] > 0:
        test.loc[i, 'numFam'] = 'pequena'
    else:
        test.loc[i, 'numFam'] = 'sem'

test = test.drop(['SibSp', 'Parch'], axis=1)




# Embarked - imputacao dos valores ausentes
mode_embarked = train['Embarked'].dropna().mode().values

train['Embarked'].fillna(mode_embarked[0], inplace=True)
test['Embarked'].fillna(mode_embarked[0], inplace=True)


# Fare
mode_embarked = train['Fare'].dropna().mode().values

train['Fare'].fillna(mode_embarked[0], inplace=True)
test['Fare'].fillna(mode_embarked[0], inplace=True)




fare = []
c1 = train['Fare'][train['Pclass'] == 1]
c2 = train['Fare'][train['Pclass'] == 2]
c3 = train['Fare'][train['Pclass'] == 3]

for i in range(len(train['Fare'])):
    if train['Fare'][i] in c1:
        fare.append("hFare")
    elif train['Fare'][i] >= 74:
        fare.append("hFare")
    elif train['Fare'][i] in c2:
        fare.append("mFare")
    elif (train['Fare'][i] < 74) and (train['Fare'][i] >= 70):
        fare.append("mFare")
    else:
        fare.append("lFare")

# print fare

train['Classe'] = fare

fare = []
c1 = test['Fare'][test['Pclass'] == 1]
c2 = test['Fare'][test['Pclass'] == 2]
c3 = test['Fare'][test['Pclass'] == 3]

for i in range(len(test['Fare'])):
    if test['Fare'][i] in c1:
        fare.append("hFare")
    elif test['Fare'][i] >= 74:
        fare.append("hFare")
    elif test['Fare'][i] in c2:
        fare.append("mFare")
    elif (test['Fare'][i] < 74) and (test['Fare'][i] >= 70):
        fare.append("mFare")
    else:
        fare.append("lFare")

test['Classe'] = fare













# Sex , Embarked - convert to numeric values
dummies = []
cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'numFam', 'Classe']
for col in cols:
    dummies.append(pd.get_dummies(train[col]))
titanic_dummies = pd.concat(dummies, axis=1)
train = pd.concat((train, titanic_dummies), axis=1)
train = train.drop(['Sex', 'Embarked','Title', 'AgeGroup', 'numFam', 'Classe'], axis=1)

dummies = []
cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'numFam', 'Classe']
for col in cols:
    dummies.append(pd.get_dummies(test[col]))
titanic_dummies = pd.concat(dummies, axis=1)
test = pd.concat((test, titanic_dummies), axis=1)
test = test.drop(['Sex', 'Embarked', 'Title', 'AgeGroup', 'numFam', 'Classe'], axis=1)


for i in range(9):
    if i not in test.columns:
        test.insert(1, i, 0)
    if i not in train.columns:
        train.insert(1, i, 0)

ageGroup = ['child', 'adolescent', 'adult', 'old', 'unknown']

for group in ageGroup:
    if group not in test.columns:
        test.insert(1, group, 0)
    if group not in train.columns:
        train.insert(1, group, 0)


# removendo atributos:
idTrain = train['PassengerId']
idTest = test['PassengerId']
trainSurvived = train['Survived']

# train = train.reindex_axis(sorted(train.columns), axis=1)
# test = test.reindex_axis(sorted(test.columns), axis=1)

train = train.drop(['Survived', 'Ticket', 'Cabin', 'Name', 'PassengerId', 'Age', 'Pclass', 'Fare'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'Age', 'Pclass', 'Fare'], axis=1)



testValues = np.empty((len(test['male']), len(test.columns)))
i = 0
for label in train.columns:
    testValues[:,i] = (test[label].values)
    i += 1


# print '###############  Testing Classifiers ################\n\n'

# #
# # print train.info()
# # print train.values[0,:]
# #
# #
# # print testValues[0,:]



# x = train.values

# y = trainSurvived.values






# def scoreTable(model, x, y):
#     ini = time.time()
#     predicted = cross_validation.cross_val_predict(model, x, y, cv=cross_validation.LeaveOneOut(891))
#     print 'Label       Precision          Recall                F1          Support'
#     precision0, recall0, fscore0, support0 = metrics.precision_recall_fscore_support(y, predicted, labels=[0])
#     print '0          ', precision0, '   ', recall0, '   ', fscore0, '   ', support0
#     precision1, recall1, fscore1, support1 = metrics.precision_recall_fscore_support(y, predicted, labels=[1])
#     print '1          ', precision1, '   ', recall1, '   ', fscore1, '   ', support1
#     print 'avg/total  ', (precision0 + precision1) / 2.0, '   ', (recall0 + recall1) / 2.0, '   ', (
#                                                                                                       fscore0 + fscore1) / 2.0, '   ', support0 + support1
#     print 'Time: ', time.time() - ini


# #Logistic Regression
# print '\n\n          Logistic Regression - Cross Validation (LeaveOneOut)'
# logisticR = LogisticRegression()
# scoreTable(logisticR,x,y)


# # Decision Tree
# print '\n\n          DecisionTree(Gini) - Cross Validation (LeaveOneOut)\n'
# decisionT = tree.DecisionTreeClassifier(criterion='gini')
# scoreTable(decisionT, x, y)
# #


# #SVM
# print '\n\n          SVM - Cross Validation (LeaveOneOut)\n'
# svmModel = svm.SVC()
# scoreTable(svmModel,x,y)


# #Naive Bayes - Gaussian
# print '\n\n          Naive Bayes - Gaussian - Cross Validation (LeaveOneOut)\n'
# naiveB = GaussianNB()
# scoreTable(naiveB,x,y)

# #KNN
# print '\n\n          KNN (5 neighbors) - Cross Validation (LeaveOneOut)\n'
# knnModel = KNeighborsClassifier(n_neighbors=5)
# scoreTable(knnModel,x,y)


# #K-Means
# print '\n\n          K-Means - Cross Validation (LeaveOneOut)\n'
# kMeans = KMeans()
# scoreTable(kMeans,x,y)


# #Random Forest
# print '\n\n          Random Forest - Cross Validation (LeaveOneOut)\n'
# randomF = RandomForestClassifier()
# scoreTable(randomF,x,y)


# #Gradient Boosting
# print '\n\n          Gradient Boosting - Cross Validation (LeaveOneOut)\n'
# gradientB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# scoreTable(gradientB,x,y)


##Submeter

x = train.values
answer = trainSurvived.values

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(x, answer)
test_result = clf.predict(testValues)

output = np.column_stack((idTest,test_result))

df_results = pd.DataFrame(output.astype('int'),columns=['PassengerId','Survived'])
df_results.to_csv('titanic_results.csv',index=False)
