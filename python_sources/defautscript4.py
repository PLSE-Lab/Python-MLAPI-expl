# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/train.csv')
#train_age_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

# preview the data
train_df.head()

train_df.tail()

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

train_df.describe(include=['O'])

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    

    
#print(combine)    
#for x in range(0, 1310):    
#    for dataset in combine:
#        text = dataset['Name']
#        textx = text[x]

#textx = textx.replace(',', 'XXXXX')
#    textx = textx.split(',', 1)[0]
#    print(textx)

#print(combine)

train_df.head()

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
#train_df = train_df.drop(['PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))
guess_ages
#print("NEWAGE")
#print(combine)

#del test_df[:]
#print("DELETE")
#print(test_df)

#train_age_df = train_df
train_df.head()
# Delete all rows with a blank age value from train_df.  Delete all rows with an age
# value from train_age_df.
#for dataset in train_df:
#    dataset.loc[ (dataset.Age.isnull()), 'Age']


#train_df = train_df[train_df.Age.isnull() != True]
#print("Whatever")
#print(train_df)
#train_age_df = train_age_df[train_age_df.Age.isnull()]
#print("FY")
#print(train_age_df)
#train_df.shape, test_df.shape

#for x in range(0, 891):
#    if np.isnan(train_df['Age'][x]):
#        train_df.drop(train_df.index[x])
#    else:
#        del train_age_df['Age'][x]

#print("TRAIN")
#print(train_df)
#print("TRAINAGE")
#print(train_age_df)


#for dataset in combine:
#    if dataset['Age'] == null:
#        print("HI")

#TEMP
combine = [train_df, test_df]

#for dataset in combine:
#    for i in range(0, 2):
#        for j in range(0, 3):
#            guess_df = dataset[(dataset['Sex'] == i) & \
#                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

#            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
#            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
#    for i in range(0, 2):
#        for j in range(0, 3):
#            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
#                    'Age'] = guess_ages[i,j]

#    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)





freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)

test_df.head(10)

# This code removes the null ages from the training set.
train_age_df = train_df
train_df = train_df[train_df.Age.isnull() != True]
train_age_df = train_age_df[train_age_df.Age.isnull()]
train_df['Age'] = train_df['Age'].astype(int)
print("TERERAE", train_df)
# This code removes the null ages from the test set.
test_age_df = test_df
test_df = test_df[test_df.Age.isnull() != True]
test_age_df = test_age_df[test_age_df.Age.isnull()]
#test_df['Age'] = test_df['Age'].astype(int)

# Build the training and test training sets.
X_train = train_df.drop("Age", axis=1)
X_train = X_train.drop("Age*Class", axis=1)
Y_train = train_df["Age"]
X_trainage = train_age_df.drop("Age", axis=1)
X_trainage = X_trainage.drop("Age*Class", axis=1)

# Build the test and test age test sets.
X_test = test_df.drop("Age", axis=1)
X_test = X_test.drop("Age*Class", axis=1)
#X_test = X_test.drop("PassengerId", axis=1)
Y_test = test_df["Age"]
X_testage = test_age_df.drop("Age", axis=1)
X_testage = X_testage.drop("Age*Class", axis=1)
#X_testage = X_testage.drop("PassengerID", axis=1)
print("TETESTS", X_test)
# Run the decision tree against the age.
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
print("AFTER FIT")
DTY_pred_train = decision_tree.predict(X_trainage)
acc_decision_tree_train = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree_train)

print("MADE IT HERE")

# Success!  Now we need to do this against the test set.
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_test, Y_test)
print("AFTER FIT")
DTY_pred_test = decision_tree.predict(X_testage)
acc_decision_tree_train = round(decision_tree.score(X_test, Y_test) * 100, 2)
print(acc_decision_tree_train)

# Done building the age data.  Now we need to reintegrate it.

newtrain_df = pd.DataFrame({
        "Survived": train_age_df["Survived"],
        "Pclass": train_age_df["Pclass"],
        "Sex": train_age_df["Sex"],
        "Age": DTY_pred_train,
        "Fare": train_age_df["Fare"],
        "Embarked": train_age_df["Embarked"],
        "Title": train_age_df["Title"],
        "IsAlone": train_age_df["IsAlone"],
        })

newtest_df = pd.DataFrame({
        "Pclass": test_age_df["Pclass"],
        "Sex": test_age_df["Sex"],
        "Age": DTY_pred_test,
        "Fare": test_age_df["Fare"],
        "Embarked": test_age_df["Embarked"],
        "Title": test_age_df["Title"],
        "IsAlone": test_age_df["IsAlone"],
        })    


newtrainwith_df = train_df
newtrainwith_df = train_df.drop("Age*Class", axis=1)
newtestwith_df = test_df
newtestwith_df = test_df.drop("Age*Class", axis=1)

combintrain = [newtrainwith_df, newtrain_df]
combintest = [newtestwith_df, newtest_df]

# Now that the data is recombined, run the learning algorithm against the survival.
X_train = newtrainwith_df.drop("Survived", axis=1)
Y_train = newtrainwith_df["Survived"]
#X_test  = test_df.drop("PassengerId", axis=1).copy()
X_test = newtestwith_df
X_train.shape, Y_train.shape, X_test.shape


#print("NWEW", newtrain_df)
#for dataset in newtrain_df:
#    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']

#print("NWEW", newtrain_df)

#for dataset in combine:
#    dataset['Age*Class'] = dataset.Age * dataset.Pclass
#submission = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
#        "Survived": RFY_pred
#    })

#submission.to_csv('submission.csv', index=False)


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
DTY_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

print("If you are here you are done.")


submissionMachLearnAge = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": DTY_pred
    })
    
submissionMachLearnAge.to_csv("submissionMLA.csv", index=False)


#print("XTrain")
#print(X_train)
#print("XTest")
#print(X_test)

# Random Forest
X_train1 = train_df.drop("Survived", axis=1)
Y_train1 = train_df["Survived"]
X_test1  = test_df.drop("PassengerId", axis=1).copy()
#X_test1 = test_df
X_train1.shape, Y_train1.shape, X_test1.shape

#combi = X_train1
#combi.append(X_test1)
#print("COMBI")
#print(combi)
#combo = [X_train1, X_test1]
#print(combo)
#combo1 = combo.drop("PassengerId", axis=1)
#X_train1 = X_train1.drop(['PassengerId'], axis=1)
#print(X_train1)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train1, Y_train1)
RFY_pred = random_forest.predict(combi)
print("THERE")
#RFY_pred = random_forest.predict(combo1)

submissionRT = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
        "Survived": RFY_pred
    })
    
submissionRT.to_csv("submissionRT.csv", index=False)






# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
LRY_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
SVMY_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
KNNY_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
GNBY_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
PERCY_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
LSVCY_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
SGDY_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
DTY_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest
X_train1 = train_df.drop("Survived", axis=1)
Y_train1 = train_df["Survived"]
#X_test1  = test_df.drop("PassengerId", axis=1).copy()
X_test1 = test_df
X_train1.shape, Y_train1.shape, X_test1.shape

combo = [X_train1, X_test1]
print(combo)
combo1 = combo.drop("PassengerId", axis=1).copy()
X_train1 = X_train1.drop(['PassengerId'], axis=1)
print(X_train1)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train1, Y_train1)
#RFY_pred = random_forest.predict(X_test)
print("THERE")
RFY_pred = random_forest.predict(combo1)
#random_forest.score(X_train, Y_train)
#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#print(acc_random_forest)
print("EVERYWHERE")
#models = pd.DataFrame({
#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#              'Random Forest', 'Naive Bayes', 'Perceptron', 
#              'Stochastic Gradient Decent', 'Linear SVC', 
#              'Decision Tree'],
#    'Score': [acc_svc, acc_knn, acc_log, 
#              acc_random_forest, acc_gaussian, acc_perceptron, 
#              acc_sgd, acc_linear_svc, acc_decision_tree]})
#models.sort_values(by='Score', ascending=False)
combine = [train_df, test_df]
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": RFY_pred
    })

#submission.to_csv('submission.csv', index=False)

#submission = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
#        "LogisticReg_Survived": LRY_pred,
#        "SVM_Survived": SVMY_pred,
#        "KNN_Survived": KNNY_pred,
#        "GaussianNaiveBayes_Survived": GNBY_pred,
#        "Perceptron_Survived": PERCY_pred,
#        "LinearSVC_Survived": LSVCY_pred,
#        "StochasticGradient_Survived": SGDY_pred,
#        "DecisionTree_Survived": DTY_pred,
#        "RandomForest_Survived": RFY_pred
#    })
    
#Store the resulting data in an output csv.
#submission.to_csv('ComparativePredictionScript4.csv', index=False)

#We now have estimates from each learning method.  Maybe we can cross reference these.
#train2_df = submission

#combine2 = [train_df, test_df]
#print(combine2)


Testo = LRY_pred + SVMY_pred + KNNY_pred + GNBY_pred + PERCY_pred + LSVCY_pred + SGDY_pred + DTY_pred + RFY_pred
#submission2 = pd.DataFrame({
#    "PassengerId": test_df["PassengerId"],
#    "Summation": Testo
#    })
   

#Build Majority Rule.
for x in range(0, 418):
    if Testo[x] >= 5:
        Testo[x] = 1
    else:
        Testo[x] = 0
        
        
submissionMajority = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Testo
    })
    
submissionMajority.to_csv("MajorityRuleTest.csv", index=False)
#submissionMajority.to_csv("submission.csv", index=False)

#Build Weighted.
WeightedTest = LRY_pred + SVMY_pred + KNNY_pred + GNBY_pred + PERCY_pred + LSVCY_pred + SGDY_pred + DTY_pred + RFY_pred

for xx in range(0, 418):
    if LRY_pred[xx] == 0:
        LRY_Weight = acc_log * -1
    else:
        LRY_Weight = acc_log
    if SVMY_pred[xx] == 0:
        SVMY_Weight = acc_svc * -1
    else:
        SVMY_Weight = acc_svc
    if KNNY_pred[xx] == 0:
        KNNY_Weight = acc_knn * -1
    else:
        KNNY_Weight = acc_knn
    if GNBY_pred[xx] == 0:
        GNBY_Weight = acc_gaussian * -1
    else:
        GNBY_Weight = acc_gaussian
    if PERCY_pred[xx] == 0:
        PERCY_Weight = acc_perceptron * -1
    else:
        PERCY_Weight = acc_perceptron
    if LSVCY_pred[xx] == 0:
        LSVCY_Weight = acc_linear_svc * -1
    else:
        LSVCY_Weight = acc_linear_svc
    if SGDY_pred[xx] == 0:
        SGDY_Weight = acc_sgd * -1
    else:
        SGDY_Weight = acc_sgd
    if DTY_pred[xx] == 0:
        DTY_Weight = acc_decision_tree * -1
    else:
        DTY_Weight = acc_decision_tree
    if RFY_pred[xx] == 0:
        RFY_Weight = acc_random_forest * -1
    else:
        RFY_Weight = acc_random_forest
        
    Weighted = LRY_Weight + SVMY_Weight + KNNY_Weight + GNBY_Weight + PERCY_Weight + LSVCY_Weight + SGDY_Weight + DTY_Weight + RFY_Weight
    
    if Weighted > 0:
        WeightedTest[xx] = 1
    else:
        WeightedTest[xx] = 0
        
submissionWeighted = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": WeightedTest
    })
    
submissionWeighted.to_csv("WeightedRuleTest.csv", index=False)
        
#for testo in Testo:    
#    if testo >= 5:
#        testo = 1
#        Testo = testo
#    else:
#        testo = 0
        

        
presubmission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Testo,
        "Pclass": test_df["Pclass"],
        "Sex": test_df["Sex"],
        "Age": test_df["Age"],
        "Fare": test_df["Fare"],
        "Embarked": test_df["Embarked"],
        "Title": test_df["Title"],
        "IsAlone": test_df["IsAlone"],
        "Age*Class": test_df["Age*Class"]
    })
print("JI")   
#presubmission.to_csv('ComparativePreditionScript4a.csv', index=False)
print("JJKKK")  
#print(submission2)
test_Rf = test_df

test_Rf['PassengerId'] = test_df['PassengerId']
test_Rf['Survived'] = Testo

#print(test_df)

#test_Qf = pd.read_csv('ComparativePredictionScript4.csv')
#print(test_Qf)

#combo = [train_df, test_Rf]
#print(combo)

#for dataset in test_df:
#    Testp = dataset['SummedResults']
    
#    if dataset.SummedResults == 9:
#        print("A 9")
#    if dataset['PassengerId'] == 9:
#        print("A 9")
#    if dataset['SummedResults'] == 0:
#        print("A 0")

#for subby in submission2:
#    print(subby)

#for LRY in LRY_pred:
#    print(LRY)


#for dataset in submission:
#    if (dataset[1] + dataset[2]) == 2:
#        print("Its a 2")
#    else:
#        print(dataset.LogisticReg_Survived.str)
#        print(dataset['LogisticReg_Survived'])
#        print(dataset[1])
#        print(dataset[2])
#        print("Its not")
    
#    dataset['LogisticReg_Survived'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
