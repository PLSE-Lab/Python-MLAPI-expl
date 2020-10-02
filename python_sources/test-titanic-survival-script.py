# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
full = train.append(test, ignore_index = True)

del train, test

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(full.head())

print("\n\nSummary statistics of training data")
print(full.describe())

#Any files you save will be available in the output tab below
full.to_csv('copy_of_the_training_data.csv', index=False)

sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
print(sex.head())

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
print(embarked.head())

# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
print(pclass.head())

# Create dataset
imputed = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
imputed['Age'] = full.Age.fillna(-0.5)
bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
group_names = [0, 1, 2, 3, 4, 5, 6, 7]
categories = pd.cut(imputed['Age'], bins, labels=group_names)
imputed['Age'] = categories

# Fill missing values of Fare with the average of Fare (mean)
#imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )
imputed[ 'Fare' ] = full.Fare.fillna(-0.5)
bins = (-1, 0, 8, 15, 31, 1000)
group_names = [0, 1, 2, 3, 4]
categories = pd.cut(imputed[ 'Fare' ], bins, labels=group_names)
imputed[ 'Fare' ] = categories

print(imputed.head())

title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

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

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
print(title.head())

cabin = pd.DataFrame()

# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

print(cabin.head())

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
print(ticket.head())

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1


# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
family.drop('FamilySize',1)
print(family.head())

# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket, title

full_X = pd.concat( [ imputed, sex, pclass, family] , axis=1 )
print(full_X.head())

train_valid_X = full_X[0:891]
train_valid_y = full[0:891].Survived

test_X = full_X[891:]

num_test = 0.20
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , test_size=num_test, random_state=23 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)


model = KNeighborsClassifier(n_neighbors = 3)

print(model.fit(train_X, train_y))

# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

test_y = model.predict(test_X)
passengerId = full[891:].PassengerId
pd.options.display.float_format = '{:,.0f}'.format
test = pd.DataFrame( { 'PassengerId': passengerId , 'Survived': test_y } )
print(test.shape)
print(test.head())
test.to_csv( 'titanic_pred.csv', index = False, float_format='%.f' )
