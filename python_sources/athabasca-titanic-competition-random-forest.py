#!/usr/bin/python
"""
Started on    March 17.2017
Completed on  March 21. 2017
@author: Jamie de Domenico and Alex Solter

This code is used to evaluate a csv file with data of the 
surviors and non survors of the trgic Titanic sinking in April 14 - 15, 1912.
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
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import sys





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
def get_combined_data(noage):
    global train_rows
    global train0

    # reading train data
    train = pd.read_csv('../input/train.csv')

    # reading test data
    test = pd.read_csv('../input/test.csv')
    
    if (noage):
        train = train[pd.notnull(train['Age'])]
        test = test[pd.notnull(test['Age'])]
        
    train0 = train.copy()
    
    train_rows = train.shape[0] - 1
    print("df_train rows = ", train_rows)  
    print("df_test rows = ", test.shape[0] - 1)

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    # this gives use a bigger dataset to work with
    combined = train.append(test)
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
  
# Method that allows you to choose the way you want 
# to process the age.
# description of the methods for choice is defined
# in the method comments
#  choice == 1  -> process_age_median
#  choice == 2  -> process_age_defined  
#  choice == 3  -> process_age_RF  
def process_age(choice): 
    if choice == 1:
        process_age_median()
    elif choice == 2:
        process_age_defined()
    else:
        process_age_median()

# Method fills in the blank ages for the missing age field 
# We use the sex, and Pclass and the title to best fit  
# with a calculated median value for the age.
# This is a feature  
def process_age_median():
    global combined
    combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    status('age median')

# Method fills in the blank ages for the missing age field 
# We use the sex, and Pclass and the title to best fit 
# the age of the passanger based on the training data.
# This is a feature
def process_age_defined():
    
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
                return 2
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
    
    status('age defined')
    
    
    # Method to fill in the blank ages using a decision tree algorithm.
    # here we are going to fill the ages in using a machine learnign algorithm to
    # give us a better approximation to the age of the passanger

def process_age_RF():
    
    global combined

    combined_age = combined.copy()
    combined_noage = combined.copy()
    
    # Here's where we use the algorithm to fill in the missing data.
    
    # This code removes the null ages from the combined set.
    # First, separate the dataset into one with ages and one without.
    # combined_age is the Dataframe with no NA's in the Age cells
    # combined_noage is the Dataframe with all the rows where Age has no value it is Nan
    combined_age.dropna(inplace=True)
    combined_noage = combined_noage.drop(combined_noage.index[combined_noage.Age > 0])
    print ("len of len(combined_age) = ", len(combined_noage)) 
    if len(combined_noage) != 0:
        
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
        
        combined_noage.Age = RFC_pred
        frames = [combined_age, combined_noage]
        combined = pd.concat(frames, ignore_index=True, axis=0)

        status('Age RF')
    

# Method to process the names.
# we are cleaning up the titles here.
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
 # Here we replace the fares with a missing median value
 # not sure how much of a feature a fare is since they are 
 # all over the place.
 # this might be another good place to use machine learning 
 # algorithm to fill in the missing fares
 # This is a feature
def process_fares():
    
    global combined
    # fill the Nan and 0 values with the Fare median Values
    combined.Fare.fillna(0,inplace=True)
    combined['Fare'].replace(0, combined.groupby('Pclass')['Fare'].median(), inplace=True)
    status('fare')

       

# Method to process the Embark point.
# We need to fill in the point of departure for 
# missing values.
# I do not think there is much value here in this field.
# but it is possible that the embark point code be 
# associated with class.
# This is a feature
def process_embarked():
    
    global combined
    # for the  missing embarked values - filling them with the most frequent one (S)
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
# this will provide 3 columns
# one for 1 -> col_1, 2 -> col_2, 3 -> col_3
# a 0 is put into the column that is associated with the Pclass
# example index 0 -> Pclass = 3
#         index 1 -> Pclass = 1
#         index 2 -> Pclass = 1
#  index  Pclass_1 Pclass_2 Pclass_3
#    0       0        0        1
#    1       1        0        0
#    2       1        0        0
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
    # this uses the prefix as the dummy column title 
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
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')
    
# Method to scale all the features.
# This is where we build the feature list.
# We remove the passanger ID from the list since it is unique
# and we cannot use it as a feature
def scale_all_features():
    
    global combined

    # combine the colums and remove the PassengerID, no need for this
    features = list(combined.columns)
    features.remove('PassengerId')
    print ('Features scaled successfully !')


# Method to compute the score
def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

# Utility Method to reurn the data sets  
# after all the processing for features
# we need to put the data sets back to it's
# original state, meaning 2 data sets
# training and testing
def recover_train_test_target():
    global combined
    global train0
    
    # we have not yet read the train data
    # train0 = pd.read_csv('../input/train.csv')
    
    # sort and re-index the data since we do not want this out of 
    # order or with an incorrect index since it will affect the [0:890] ans [891:]
    combined = combined.sort_values(by='PassengerId', ascending=1)
    combined = combined.reset_index(drop=True)
    
    # targets are the survivors
    targets = train0.Survived
    # train = combined.ix[0:890]
    train = combined.ix[0:train_rows]
    # test = combined.ix[891:]
    test = combined.ix[train_rows+1:]
    
    return train,test,targets 
    
# Method to run all the Data Preprocessing.
# this is just a utility method so that we 
# can make one call from one spot to process
# all the data.
def data_processing():
    
    global combined

    # used to set the processing of the age type
    # 1-median
    # 2-defined
    # 3-RF
    # and other value defaults to median
    age_Process_type = 3
 

    # run the get_combined_data method
    # we have a boolean flag that allows us
    # to remove rows if the Age field is empty
    # This is to test if it is better to predict
    # the possible value or remove it out right
    combined = get_combined_data(False)
    
    # run the get_titles method
    get_titles()

    # run the process_fares method
    process_fares()

    # run the process_age method
    if age_Process_type != 3:
        process_age(age_Process_type)
    
    # run the process_names method
    process_names()


    # run the process_embarked method
    process_embarked()

    # run the process_cabin method
    process_cabin()

    # run the process_sex method
    process_sex()

    # run the process_pclass method
    process_pclass()

    # run the process_ticket method
    process_ticket()

    # run the process_family method
    process_family()

    # run the process_age_RF method
    if age_Process_type == 3:
        process_age_RF()
    
"""**************************************************************************************"""
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
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)

# create a features data set and fill it
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

# print out what we have sorted so we can see it - the importance things
print (features.sort_values(['importance'],ascending=False))

# get the model from clf and train it
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
test_new = model.transform(test)

# set up the forest using sqrt
# set up the parameter grid
forest = RandomForestClassifier(max_features='sqrt')
p_grid = {
            'max_depth' : [4,5,6,7,8],
            'max_features' : ['auto',None],
            'n_estimators' : [100,200,300],
            'random_state': [7],
            'criterion': ['gini','entropy']
         }
# povides train/test indices to split data in train/test sets
cross_validation = StratifiedKFold(targets, n_folds=5)

# provides ehaustive search over specified parameter values for an estimator
grid_search = GridSearchCV(estimator=forest, param_grid=p_grid, cv=cross_validation)
grid_search.fit(train_new, targets)

# display the best scores and the parameters
print("-----------------------Scoring-----------------------------------")
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# run the prediction and output the file.
# file is located in the directory the program ran from
pipeline = grid_search
output = pipeline.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('random_forest.csv',index=False)