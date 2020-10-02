#Import relevant packages

import numpy as np #
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from scipy.stats import randint
from sklearn import preprocessing




#Import training and test data sets

titanic = pd.read_csv("../input/train.csv")

test_titanic_actual = pd.read_csv("../input/test.csv")




print ('titanic:' , titanic.shape, 'actual testing set: ', test_titanic_actual.shape)



#Use groupby to obtain information about survivial rates associated with various columns of the data frame



surv_class = titanic['Survived'].groupby(titanic['Pclass'])


print(surv_class.mean())


surv_sex = titanic['Survived'].groupby(titanic['Sex'])

print(surv_sex.mean())


surv_sibsp = titanic['Survived'].groupby(titanic['SibSp'])

print(surv_sibsp.mean())

surv_parch = titanic['Survived'].groupby(titanic['Parch'])

print(surv_parch.mean())


surv_class_embarked = titanic['Survived'].groupby([titanic['Embarked'],titanic['Pclass']])

print(surv_class_embarked.mean())


#Use Seaborn package to make visualizations of the data set



g0 = sns.FacetGrid(titanic, col='Survived')
g0.map(plt.hist, 'SibSp', bins=6)

g0.savefig("test0.png")


g4 = sns.FacetGrid(titanic, col='Survived')
g4.map(plt.hist, 'Parch', bins=6)

g4.savefig("test4.png")


g1 = sns.FacetGrid(titanic, col='Pclass')
g1.map(sns.countplot, 'Survived')
g1.add_legend()
g1.savefig("test1.png")


g2 = sns.FacetGrid(titanic, col='Pclass', row = 'Embarked')
g2.map(sns.countplot, 'Survived')
g2.add_legend()
g2.savefig("test2.png")



print(titanic['Embarked'].isnull().any())

#Fill null values with most common embarkation port

titanic['Embarked'] = titanic['Embarked'].fillna('S')


print(titanic['Embarked'].isnull().any())



#Set Male/Female to binary

titanic['Sex'] = titanic['Sex'].map(lambda x : 1 if x =='female' else 0)


#Define a new column of total family members on board for each person


titanic['FamilyMem'] = titanic['Parch'] + titanic['SibSp']



g5 = sns.FacetGrid(titanic, col='Survived')
g5.map(plt.hist, 'FamilyMem', bins=6)

g5.savefig("test5.png")

surv_fam_memb = titanic['Survived'].groupby(titanic['FamilyMem'])

print(surv_fam_memb.mean())


#Use family members to define a new feature determining if the person is traveling alone or not


titanic['Alone'] = titanic['FamilyMem'].map(lambda x : 1 if x == 0 else 0)


surv_alone = titanic['Survived'].groupby([titanic['Alone']])

print(surv_alone.mean())


#Fill in unknown empty values of Age with the median value

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())


#Make  a child feature with some age, here less than 9.5


titanic['Child'] = titanic['Age'].map(lambda Age : 1 if Age < 9.5 else 0)

#Make a family feature indicating many family members on board

titanic['Family'] = titanic['FamilyMem'].map(lambda x : 1 if x >= 4 else 0)


surv_family = titanic['Survived'].groupby([titanic['Family']])

print(surv_family.mean())



#Split embarkation into 3 new features


titanic['Cherbourg'] = (titanic['Embarked'] == 'C').astype('int')

titanic['Southampton'] = (titanic['Embarked'] == 'S').astype('int')

titanic['Queenstown'] = (titanic['Embarked'] == 'Q').astype('int')



#Train off of sex, passenger class, whether alone or not

X_titanic = titanic.drop(['Survived', 'SibSp', 'Name', 'Parch','Fare','Embarked','PassengerId','FamilyMem','Ticket','Cabin','Age'], axis=1)


print(X_titanic.head())

Y_titanic = titanic['Survived']


#Train_test_spilit on the training data

X_titanic_train, X_titanic_test, Y_titanic_train, Y_titanic_test = train_test_split(X_titanic, Y_titanic,test_size=0.2,random_state=42)



#Define a parameter distribution or use in a Decision tree classifier

param_dist = {"max_depth": [None, 2, 3, 5, 10],
              "max_features": ['log2', 'sqrt','auto'],
              "min_samples_leaf": [1,5,8],
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier called tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object called tree_cv
tree_cv = GridSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_titanic_train,Y_titanic_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best tree score is {}".format(tree_cv.best_score_))


# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate a logistic regression classifier 
logreg = LogisticRegression()

# Instantiate the GridSearchCV object
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X_titanic_train,Y_titanic_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best logreg score is {}".format(logreg_cv.best_score_))




#########################################################3
#Setup the features on the actual test data set



test_titanic_actual['Age'] = test_titanic_actual['Age'].fillna(test_titanic_actual['Age'].median())
test_titanic_actual['Sex'] = test_titanic_actual['Sex'].map(lambda x : 1 if x =='female' else 0)


test_titanic_actual['FamilyMem'] = test_titanic_actual['Parch'] + titanic['SibSp']

test_titanic_actual['Alone'] = test_titanic_actual['FamilyMem'].map(lambda x : 1 if x == 0 else 0)

test_titanic_actual['Child'] = test_titanic_actual['Age'].map(lambda Age : 1 if Age < 9.5 else 0)
test_titanic_actual['Family'] = test_titanic_actual['FamilyMem'].map(lambda x : 1 if x >= 4 else 0)


test_titanic_actual['Cherbourg'] = (test_titanic_actual['Embarked'] == 'C').astype('int')

test_titanic_actual['Southampton'] = (test_titanic_actual['Embarked'] == 'S').astype('int')

test_titanic_actual['Queenstown'] = (test_titanic_actual['Embarked'] == 'Q').astype('int')



X = test_titanic_actual.drop(['SibSp', 'Name', 'Parch','Fare','FamilyMem','Embarked','PassengerId','Ticket','Cabin','Age'], axis=1)


print(X.head())





#Best Decision Tree Fit

tree_2 = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features = 'log2', min_samples_leaf = 1)

tree_2.fit(X_titanic_train, Y_titanic_train)

Y_2 = tree_2.predict(X)



sol_2  = pd.DataFrame({
        "PassengerId": test_titanic_actual["PassengerId"],
        "Survived": Y_2
    })

sol_2.to_csv('sex_class_age_alone_family_tree.csv',index=False)


