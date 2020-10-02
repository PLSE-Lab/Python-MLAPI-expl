import pandas as pd 
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
######################################################################
########
# Replace NaNs in the 'Age' column with randomly generated values from a truncated norm
########
import scipy.stats as stats

# column 'Age' has some NaN values
# A simple approximation of the distribution of ages is a gaussian, but this is not commonly accurate.
# lets make a vector of random ages centered on the mean, with a width of the std
lower, upper = train_df['Age'].min(), train_df['Age'].max()
mu, sigma = train_df["Age"].mean(), train_df["Age"].std()

# number of rows
n = train_df.shape[0]

# vector of random values using the truncated normal distribution.  
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
rands = X.rvs(n)

# get the indexes of the elements in the original array that are NaN
idx = np.isfinite(train_df['Age'])

# use the indexes to replace the NON-NaNs in the random array with the good values from the original array
rands[idx.values] = train_df[idx]['Age'].values

## At this point rands is now the cleaned column of data we wanted, so push it in to the original df
train_df['Age'] = rands

#### Do the same for the test data
# column 'Age' has some NaN values
# A simple approximation of the distribution of ages is a gaussian, but this is not commonly accurate.
# lets make a vector of random ages centered on the mean, with a width of the std
lower, upper = test_df['Age'].min(), test_df['Age'].max()
mu, sigma = test_df["Age"].mean(), test_df["Age"].std()

# number of rows
n = test_df.shape[0]

# vector of random values using the truncated normal distribution.  
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
rands = X.rvs(n)

# get the indexes of the elements in the original array that are NaN
idx = np.isfinite(test_df['Age'])

# use the indexes to replace the NON-NaNs in the random array with the good values from the original array
rands[idx.values] = test_df[idx]['Age'].values

## At this point rands is now the cleaned column of data we wanted, so push it in to the original df
test_df['Age'] = rands
######################
######################################################################

##########
# Cabin

# Code based on that here: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# replacing missing cabins with U (for Uknown)
train_df.Cabin.fillna('U',inplace=True)
# mapping each Cabin value with the cabin letter
train_df['Cabin'] = train_df['Cabin'].map(lambda c : c[0])
# dummy encoding ...
cabin_dummies = pd.get_dummies(train_df['Cabin'],prefix='Cabin')
train_df = pd.concat([train_df,cabin_dummies],axis=1)
train_df.drop('Cabin',axis=1,inplace=True)

# replacing missing cabins with U (for Uknown)
test_df.Cabin.fillna('U',inplace=True)
# mapping each Cabin value with the cabin letter
test_df['Cabin'] = test_df['Cabin'].map(lambda c : c[0])
# dummy encoding ...
cabin_dummies = pd.get_dummies(test_df['Cabin'],prefix='Cabin')
test_df = pd.concat([test_df,cabin_dummies],axis=1)
test_df.drop('Cabin',axis=1,inplace=True)

######################
######################################################################

##########
# Family

# Code based on that here: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# introducing a new feature : the size of families (including the passenger)
train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp'] + 1
# introducing other features based on the family size
train_df['Singleton'] = train_df['FamilySize'].map(lambda s : 1 if s == 1 else 0)
train_df['SmallFamily'] = train_df['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
train_df['LargeFamily'] = train_df['FamilySize'].map(lambda s : 1 if 5<=s else 0)

# Code based on that here: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# introducing a new feature : the size of families (including the passenger)
test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp'] + 1
# introducing other features based on the family size
test_df['Singleton'] = test_df['FamilySize'].map(lambda s : 1 if s == 1 else 0)
test_df['SmallFamily'] = test_df['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
test_df['LargeFamily'] = test_df['FamilySize'].map(lambda s : 1 if 5<=s else 0)
########
######################################################################

###############
# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
train_df['Person'] = train_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
train_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column
person_dummies_titanic  = pd.get_dummies(train_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
#person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
#person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_df = train_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

train_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)
###############
######################################################################

############
# Pclass

# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_train  = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

train_df = train_df.join(pclass_dummies_train)
test_df    = test_df.join(pclass_dummies_test)
##############
######################################################################


############
# Ticket
# Code based on that here: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip() , ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
    
train_df['Ticket'] = train_df['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(train_df['Ticket'],prefix='Ticket')
train_df = pd.concat([train_df, tickets_dummies],axis=1)
train_df.drop('Ticket',inplace=True,axis=1)

test_df['Ticket'] = test_df['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(test_df['Ticket'],prefix='Ticket')
test_df = pd.concat([test_df, tickets_dummies],axis=1)
test_df.drop('Ticket',inplace=True,axis=1)
##############
######################################################################


############
# Title
# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
# we extract the title from each name
train_df['Title'] = train_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
# we map each title
train_df['Title'] = train_df.Title.map(Title_Dictionary)
# we extract the title from each name
test_df['Title'] = test_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
# we map each title
test_df['Title'] = test_df.Title.map(Title_Dictionary)
# encoding in dummy variable
titles_dummies = pd.get_dummies(train_df['Title'],prefix='Title')
train_df = pd.concat([train_df,titles_dummies],axis=1)
titles_dummies = pd.get_dummies(test_df['Title'],prefix='Title')
test_df = pd.concat([test_df,titles_dummies],axis=1)
# removing the title variable
train_df.drop('Title',axis=1,inplace=True)
test_df.drop('Title',axis=1,inplace=True)
##############
######################################################################



##############
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
train_df["Embarked"] = train_df["Embarked"].fillna("S")
# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_train  = pd.get_dummies(train_df['Embarked'])
#embark_dummies_train.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
#embark_dummies_test.drop(['S'], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_train)
test_df    = test_df.join(embark_dummies_test)

train_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)
##############
######################################################################
######################################################################

#######
# Drop all extra columns that have no effect and/or confuse the fitting
#####
#train_df.drop("Cabin",axis=1,inplace=True)
#test_df.drop("Cabin",axis=1,inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
train_df.drop(['Name'], axis=1,inplace=True)
test_df.drop(['Name'], axis=1,inplace=True)
#train_df.drop(['Ticket'], axis=1,inplace=True)
#test_df.drop(['Ticket'], axis=1,inplace=True)
#train_df.drop(['PassengerId'], axis=1,inplace=True)
######################

######################################################################
######################################################################
# Scale all features except passengerID
features = list(train_df.columns)
features.remove('PassengerId')
train_df[features] = train_df[features].apply(lambda x: x/x.max(), axis=0)

features = list(test_df.columns)
features.remove('PassengerId')
test_df[features] = test_df[features].apply(lambda x: x/x.max(), axis=0)
######################################################################
######################################################################

######################################################################
######################################################################
## Remove extra columns in training DF that are not in test DF
train_cs = list(train_df.columns)
train_cs.remove('Survived')
test_cs = list(test_df.columns)
for c in train_cs:
    if c not in test_cs:
        #print repr(c)+' not in test columns, so removing it from training df'
        train_df.drop([c], axis=1,inplace=True)
for c in test_cs:
    if c not in train_cs:
        #print repr(c)+' not in training columns, so removing it from test df'
        test_df.drop([c], axis=1,inplace=True)
######################################################################
######################################################################

#######
# define training and testing sets
########
X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.copy()
##########

######################################################################
######################################################################
# Feature selection
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train)
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)

model = SelectFromModel(clf, prefit=True)
X_train_new = model.transform(X_train)
X_train_new.shape

X_test_new = model.transform(X_test)
X_test_new.shape
######################################################################
######################################################################


###########
# Random Forests
##########
random_forest = RandomForestClassifier(n_estimators=270)
random_forest.fit(X_train_new, Y_train)
Y_pred = random_forest.predict(X_test_new)
print('standard score ', random_forest.score(X_train_new, Y_train))
print('cv score ',np.mean(cross_val_score(random_forest, X_train_new, Y_train, cv=10)))
###########


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)

























