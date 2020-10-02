#!/usr/bin/python
"""
Started on    March 17.2017
COmpleted on  March xx. 2017
@author: Jamie de Domenico and Alex Solter

This code is used to evaluate a csv file with data of the 
surviors and non survors of the trgic Titanic sinking in april 14 - 15, 1912.
This is a prediction model that will output the roster of a training set
of the passengers by passanger number of who perished and who survived.
A value of ZERO means the passanger did not survive.
A value of ONE means the passange survived

"""


# import libraries
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import cross_val_score

import pandas as pd
import numpy as np
import sys
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


# remove warnings
# warnings.filterwarnings('ignore')




# Method to print the that we are processing a feature.
# I use this for ouput debugging so I know there 
# is a feature being processed.
def status(feature):
    print ("Processing",feature,": ok")
    
# Read in the files and combine the data
# this gives us a bigger set of data to look at 
# for processing.
# We train on one and test on a combination of the 2 files.
# We will remove suvived from the train data hen we test.
def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    # test = train
    
    train.to_csv('test_no_survivor_col.csv',index=False)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    return combined
    
# Read in the files and combine the data
# this gives us a bigger set of data to look at 
# for processing.
# We train on one and test on a combination of the 2 files.
# We will remove suvived from the train data hen we test.
def combined_drop_column(col):

    global combined
    combined.drop(col,axis=1,inplace=True)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    return combined    
    
    
# Method to create the titles for each passenger.
# We use this method to give us a better feature
# for when we are running the training and testing.
def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Lady" :      "Royalty",
                        "Sir" :       "Royalty",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",        
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",        
                        "Dr":         "Officer",
                        "Mme":        "Mrs",
                        "Ms":         "Mrs",
                        "Mrs" :       "Mrs",
                        "Mlle":       "Miss",
                        "Miss" :      "Miss",
                        "Master" :    "Master",        
                        "Mr" :        "Mr",
                        "Rev":        "Reverend"
                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

# Method fills in the blank ages for the missing age field 
# We use the sex, and Pclass and the title to best fit 
# the age of the passanger based on the training data.
# This is a feature
def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39 
                

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30
       

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31
       

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40
        

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5
            elif row['Title'] == 'Reverend':
                return 43                 

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
    

# Method to fill in the blank ages using a decision tree algorithm.
def process_age_DT():
    
    global combined
    
    # Here's where we use the algorithm to fill in the missing data.
    
    # This code removes the null ages from the combined set.
    # First, separate the dataset into one with ages and one without.
    combined_age = combined
    combined_noage = combined
    combined_age = combined_age[combined_age.Age.isnull() != True]
    combined_noage = combined_noage[combined_noage.Age.isnull()]
    combined_age["Age"] = combined_age["Age"].astype(int)
    # Next, drop the age field from each set.  This is going to be used as
    # the target field for the learning algorithm.
    X_combined_age = combined_age.drop("Age", axis=1)
    X_combined_age = X_combined_age.drop("Fare", axis=1)
    X_combined_age = X_combined_age.drop("PassengerId", axis=1)
    Y_combined_age = combined_age["Age"].astype(int)
    X_combined_noage = combined_noage.drop("Age", axis=1)
    X_combined_noage = X_combined_noage.drop("Fare", axis=1)
    X_combined_noage = X_combined_noage.drop("PassengerId", axis=1)

    # Run the random forest against the age.
    RFC = RandomForestClassifier()
    RFC.fit(X_combined_age, Y_combined_age)
    RFC_pred = RFC.predict(X_combined_noage)
    acc_RFC = round(RFC.score(X_combined_age, Y_combined_age) * 100, 2)
    
    print("---------------------------")
    print(acc_RFC)
    print("__________________________")

    # Reintegrate the age data back into the dataset.
    new_combined_noage = pd.DataFrame({
        "PassengerId": combined_noage["PassengerId"],
        "Sex": combined_noage["Sex"],
        "Age": RFC_pred,
        "SibSp": combined_noage["SibSp"],
        "Parch": combined_noage["Parch"],
        "Fare": combined_noage["Fare"],
        "Title_Master": combined_noage["Title_Master"],
        "Title_Miss": combined_noage["Title_Miss"],
        "Title_Mr": combined_noage["Title_Mr"],
        "Title_Mrs": combined_noage["Title_Mrs"],
        "Ticket_STONO2": combined_noage["Ticket_STONO2"],
        "Ticket_STONOQ": combined_noage["Ticket_STONOQ"],
        "Ticket_SWPP": combined_noage["Ticket_SWPP"],
        "Ticket_WC": combined_noage["Ticket_WC"],
        "Ticket_WEP": combined_noage["Ticket_WEP"],
        "Ticket_XXX": combined_noage["Ticket_XXX"],
        "FamilySize": combined_noage["FamilySize"],
        "Singleton": combined_noage["Singleton"],
        "SmallFamily": combined_noage["SmallFamily"],
        "LargeFamily": combined_noage["LargeFamily"],
        })
    
    
#    combined_noage.Age = RFC_pred
#    combined = [combined_age, new_combined_noage]
    print("THERE")
#    print(combined)



# Method to process the names.
# we are cleaning up the titles here for missing titles in the names.
# This is a feature
def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')
    
 # Method to process the fares
 # Here we replace the fares with a missing mean value
 # not sure how much of a feature a fare is since they are 
 # all over the place.
 # This is a feature
def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    
    status('fare')

# Method to process the Embark point.
# We need to fill in the point of departure for 
# missing values.
# This is a feature
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')

# Method to fill in the missing cabin data.
# This is filled in using a cabin prefix
# This is a feature
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
    status('cabin')
    
# Method to process the sex of the passenger
# We set the passenger from Male to 1 and Female to 0
# This makes it much easier since the sev is now numerical
# This is a feature
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    status('sex')

# Method to Process the Pclass
# here we are going to fill in the empty values to
# complete the data.
# This is a feature but I am not sure how much use there is for it
# since there is alot of missing data.
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')  
    
# Method to process the ticket for missing values.
# Here we are processing the ticket since we want complete data.
# This is not a feature since the tickets have no value or pattern.
# This is why I have entered XXX for passengers with no ticket number.
# we then convert XXX to a float 0.00000
# all ticket numbers are converted to float
def process_ticket():
    
    global combined
    
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
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')

# Method to process a family.
# This is used to catagorized/combine family members 
# We do this by size Singleton (1 member), Small(upto 4 members), LargeFamily (5 or more members)
# This is a feature.
def process_family():
    
    global combined
    
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=3 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 4<=s else 0)
    
    status('family')
    
# Method to scale all the features.
# This is where we build the feature list.
# We remove the passanger ID from the list since it is unique
# and we cannot use it as a feature
def scale_all_features():
    
    global combined
    print(combined)
    # combine the colums and remove the PassengerID, no need for this
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print ('Features scaled successfully !')
    print (combined[features])


# Method to compute the score
def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

# Utility Method to reurn the data sets    
def recover_train_test_target():
    global combined
    
    # we have not yet read the train data
    train0 = pd.read_csv('../input/train.csv')

    # targets are the survivors
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets 
    
# Method to run all the Data Preprocessing.
# this is just a utility method so that we 
# can make one call from one spot to process
# all the data.
def data_processing():
    
    global combined
    # run the get_combined_data method
    combined = get_combined_data()
    combined.shape
    combined.shape
    combined.head()
    
    # combined_drop_column('Ticket')
    # combined_drop_column('Cabin')
    # combined_drop_column('Fare')
    combined_drop_column('Embarked')
    combined.shape
    combined.shape
    combined.head()
    
    # run the get_titles method
    get_titles()
    combined.head()
    grouped = combined.groupby(['Sex','Pclass','Title', 'Parch'])
    grouped.median()
    
    # run the process_age method
    process_age()
    combined.info()
    
    # run the process_names method
    process_names()
    combined.head()
    
    # run the process_ticket method
    process_ticket() 
    
    # run the process_fares method
    process_fares()
    
    # run the process_cabin method
    process_cabin()
    

    """
    # run the process_fares method
     process_fares()
    
    # run the process_embarked method
    process_embarked()
   
    # run the process_cabin method
    process_cabin()

    # run the process_ticket method
    process_ticket()    
    """
    combined.info()
    combined.head()

    
    # run the process_sex method
    process_sex()
    
    # run the process_pclass method
    process_pclass()
    
    # run the process_ticket method
   # process_ticket()
    
    # run the process_family method
    process_family()
    
   # process_age_DT()
    combined.info()

    # Combine and shape the data
    combined.shape
    combined.head()
    
    # scale all the features
    scale_all_features()
    
    

# Method main_method
# Here is the main code
# all the action happens here
# we fist go through setting up the data and then run the 
# training and then testing

# run the main routine to process the data
data_processing()

# get the dta
train,test,targets = recover_train_test_target()

# get the classifiers 200 estimator
# we could raise this to see if we get a better result
clf = ExtraTreesClassifier(n_estimators=100)
clf = clf.fit(train, targets)
print ("train.columns training set")
print (train.columns)

print ("test.columns training set")
print (test.columns)

print ("clf.feature_importances_")
print (clf.feature_importances_)

# create a features data set and fill it
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

# print out what we have sorted so we can see it - the importance things
print (features.sort_values(['importance'],ascending=False))
features.sort_values(['importance'],ascending=False).to_csv('features_and_Importance_from_training.csv',index=False)

# get the model from clf and train it
# we print this so we can see what the outcome is
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
print(train_new.shape)

# create a test_new from the train data and shape it
test_new = model.transform(test)
test_new.shape

# set up the forest using sqrt
# set up the parameter grid
forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
                  'max_depth' : [None, 10,20],
                  'max_features' : ['auto',None],
                  'n_estimators' : [100,200,300],
                  'random_state': [7]
                 }

# povides train/test indices to split data in train/test sets
cross_validation = StratifiedKFold(targets, n_folds=5)

# provides ehaustive search over specified parameter values for an estimator
grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
grid_search.fit(train_new, targets)

# display the best scores and the parameters
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# run the prediction and output the file.
pipeline = grid_search
output = pipeline.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('random_forest.csv',index=False)
