
# # Team YSI - Titanic: Machine Learning from Disaster

# ## Version 1

#########################################################################

#

# Titanic: Machine Learning from Disaster

#

# Python script for generation of a model predicting the survivals.

#

# Amendment date             Amended by            Description

# 22/11/2016                 Ivaylo Shalev         Initial version.

# 26/11/2016                 Ivaylo Shalev         Added LR for Age missing values.

#

#

#########################################################################

import pandas as pd

import numpy as np

from sklearn import cross_validation

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier



# Reading of input data (train and test)

main_train_df = pd.read_csv('../input/train.csv', header=0)      # Load the train file into a dataframe

main_test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe



# The test data doesn't contain the target (survived), however it still can be used when we are doing data preparation

# That's why we create a third dataframe which will contain both training and test data into one.

# When executing the modeling we will split them back.

main_all_df = main_train_df.append(main_test_df)              # Create a union between both data frames



# Show some stats

print("Training data - number of rows: %s" %(main_train_df['PassengerId'].size))

print("Testing data - number of rows:  %s" %(main_test_df['PassengerId'].size))

print("Total data - number of rows:    %s" %(main_all_df['PassengerId'].size))

print("")



# training data

print("ALL DATA")

# show first row

print(main_all_df.iloc[0])

print("")

# show last row

print(main_all_df.iloc[-1])

print("")




# Data Preparation



# PassengerId - do nothing (as it is - int), but it will not be used as a feature

# Pclass - do nothing (as it is - int 1,2,3)

# SibSp - do nothing (as it is - int 1,2,3,4,5,6,7,8)

# Parch - do nothing (as it is - int 1,2,3,4,5,6,7,8)



# Survived - convert to int

main_all_df['Survived'] = main_all_df.ix[main_all_df.Survived.isnull() == False, 'Survived'].astype(np.int)



# Sex - convert it to ID (int): 0 - female, 1 - male

main_all_df['GenderId'] = [ 0 if x == 'female' else 1 for x in main_all_df['Sex'] ]



# Family Size - sum SibSp + Parch + 1

main_all_df['FamSize'] = main_all_df.SibSp + main_all_df.Parch + 1

main_all_df['FamSizeId'] = 0

main_all_df.loc[(main_all_df.FamSize > 1) & (main_all_df.FamSize <= 4), 'FamSizeId'] = 1

main_all_df.loc[(main_all_df.FamSize > 4), 'FamSizeId'] = 2





# Name - extract family name and title

# Name - Surname

main_all_df['Surname'] = main_all_df['Name'].replace("(\\,..*)", "", regex=True)

main_all_df['SurnameId'] = main_all_df['Surname'] + "_" + main_all_df['FamSize'].astype(str)



# Name - Title - group common titles and factor them all

main_all_df['Title'] = main_all_df['Name'].replace("(.*, )|(\\..*)", "", regex=True)

common_titles = [['Other', 0], ["Miss", 1], ["Mr", 2], ["Master", 3], ["Mile", 1], ["Ms", 1], ["Mme", 2]]

common_titles_dict = { title : i for title, i in common_titles }

main_all_df['TitleId'] = [ 'Other' if x not in list(common_titles_dict) else x for x in main_all_df['Title'] ]

main_all_df['TitleId'] = main_all_df['TitleId'].map( lambda x: common_titles_dict[x])





# Embarked - decode letter to ID (int)

main_all_df['EmbarkedId'] = [ 0 if np.isnan(x) else x.astype(int) for x in main_all_df['Embarked'].map(

        {

            'C': 1 # Cherbourg

         ,  'Q': 2 # Queenstown

         ,  'S': 3 # Southampton

        })]

# fill missing

main_all_df.loc[main_all_df["EmbarkedId"] == 0, 'EmbarkedId'] = 3 # S





# Child

main_all_df['Child'] = 0

main_all_df.loc[main_all_df.Age < 18, 'Child'] = 1



# Mother

main_all_df['Mother'] = 0

main_all_df.loc[  (main_all_df.Age >= 18)

                & (main_all_df.Parch > 0)

                & (main_all_df.GenderId == 0)

                & (main_all_df.Title != "Miss"), 'Mother'] = 1



# FareGroupId

main_all_df['FareGroupId'] = 0

main_all_df.loc[main_all_df.Fare.notnull(), 'FareGroupId'] = main_all_df.loc[main_all_df.Fare.notnull(), 'Fare'].astype(int)

#print main_all_df.loc[(main_all_df.Fare.notnull()) & (main_all_df.Fare == 0),'Fare']



# Cabin - extract Deck letter and convert it to ID (int)

main_all_df['DeckId'] = [ 0 if np.isnan(x) else x.astype(int) for x in main_all_df['Cabin'].str[:1].map(

        {

           #'T': 1 # Boat Deck - most top - ignore (just 1 case)

            'A': 1 # higher

         ,  'B': 2

         ,  'C': 3

         ,  'D': 4

         ,  'E': 5

         ,  'F': 6

         ,  'G': 7 # lowest deck

        })]



# fill missing

#main_all_df.groupby('DeckId').count()['PassengerId']

# create train and target df

all_deck_train_df = main_all_df[[

     'DeckId'

    ,'Pclass'

    ,'FareGroupId'

    #,'FamSizeId'

    ,'EmbarkedId'

    #,'Survived'

    #,'TitleId'

    #,'GenderId'

    #,'Child'

    #,'SibSp'

    #,'Parch'

    #,'Mother'

]]

deck_train_df = all_deck_train_df.loc[all_deck_train_df['DeckId'] != 0].copy()

deck_null_df = all_deck_train_df.loc[all_deck_train_df['DeckId'] == 0].copy()

deck_target_df = deck_train_df['DeckId'].copy()

deck_train_df.drop(['DeckId'], axis = 1, inplace=True)

deck_null_df.drop(['DeckId'], axis = 1, inplace=True)





# Linear Regression

print('Training Deck model...')

deck_train_model = RandomForestClassifier(n_estimators=100)



# Cross validation

scores = cross_validation.cross_val_score(deck_train_model

                                          ,deck_train_df

                                          ,deck_target_df

                                          ,cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# Predict

print('Predicting Deck...')

deck_train_model = deck_train_model.fit(deck_train_df, deck_target_df)

deck_train_result = deck_train_model.predict(deck_null_df)

main_all_df.loc[main_all_df['DeckId'] == 0, 'DeckId'] = deck_train_result

print('Done.')



# Age - build regression model to fill the missing age values

# create train and target df

all_age_train_df = main_all_df[[

     'Age'

    ,'Survived'

    ,'TitleId'

    ,'GenderId'

    ,'Pclass'

    ,'Child'

    ,'FareGroupId'

    ,'FamSizeId'

    ,'SibSp'

    ,'Parch'

    ,'Mother'

    #,'EmbarkedId'

    #,'DeckId'

    #,'Fare'

]]

age_train_df = all_age_train_df.loc[all_age_train_df['Age'].notnull()].copy()

age_null_df = all_age_train_df.loc[all_age_train_df['Age'].isnull()].copy()

age_target_df = age_train_df['Age'].copy()

age_train_df.drop(['Age'], axis = 1, inplace=True)

age_null_df.drop(['Age'], axis = 1, inplace=True)



# Linear Regression

print('Training Age model...')

lreg = LinearRegression()



# Cross validation

scores = cross_validation.cross_val_score(lreg

                                          ,age_train_df

                                          ,age_target_df

                                          ,cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# Predict

print('Predicting Age...')

lreg = lreg.fit(age_train_df, age_target_df)

lreg_result = lreg.predict(age_null_df)

main_all_df.loc[np.isnan(main_all_df["Age"]), 'Age'] = lreg_result

print('Done.')




# Classification



# Split into Train and Test DF

# get only the good features, ID and Target

all_good_df = main_all_df[[

     'PassengerId'

    ,'Survived'

    ,'TitleId'

    ,'GenderId'

    ,'Age'

    ,'Pclass'

    #,'FareGroupId'

    #,'DeckId'

    ,'Child'

    #,'FamSizeId'

    #,'SibSp'

    #,'EmbarkedId'

    #,'Parch'

    #,'Mother'

    #,'Fare'

]]



# Split rows into original sets

train_df = all_good_df.ix[all_good_df.PassengerId <= 891]

test_df = all_good_df.ix[all_good_df.PassengerId > 891]



# Get ID and Target

test_ids = test_df['PassengerId'].values

target_df = all_good_df.ix[all_good_df.PassengerId <= 891, 'Survived']



# Remove ID and Target columns from the datasets

train_df = train_df.drop(['PassengerId', 'Survived'], axis = 1)

test_df = test_df.drop(['PassengerId', 'Survived'], axis = 1)



# RandomForest

print('Training...')

#forest_model = RandomForestClassifier(n_estimators=100)

#forest_model = GradientBoostingClassifier(n_estimators=100)

#forest_model = ExtraTreesClassifier(n_estimators=100)

forest_model = AdaBoostClassifier(n_estimators=1000)



# Cross validation

scores = cross_validation.cross_val_score(forest_model

                                          ,train_df

                                          ,target_df

                                          ,cv=20)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





# Predict

print('Predicting...')

forest_model = forest_model.fit(train_df, target_df)

predict_output = forest_model.predict(test_df).astype(int)

results_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': predict_output})

# Save to CSV file

results_df.to_csv(path_or_buf="ysi_titanic_prediction_v4.csv", index=False)

print('Done.')