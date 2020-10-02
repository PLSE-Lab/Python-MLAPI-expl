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

SVM Linear algorithm is used from sklearn

"""


# import libraries
import pandas as pd
import numpy as np
import re
import random
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import warnings
from timeit import default_timer as timer


# remove warnings
warnings.filterwarnings('ignore')


# Import data from the csv files
training_dataframe=pd.read_csv('../input/train.csv')
testing_dataframe=pd.read_csv('../input/test.csv')
start = timer()

# Mix the training and test data to allow for easier manipulation
# Move Survived column to front of the Trianing data
cols = training_dataframe.columns.tolist()
cols=[cols[1]]+[cols[0]]+cols[2:]
training_dataframe=training_dataframe[cols]

# Add the Survived column to front of test data
testing_dataframe.insert(0,'Survived',np.nan,True)
# Mix them together
mix=pd.concat([training_dataframe,testing_dataframe]) 


# Change the data into a useful to classifier
# Remove useless data
mix=mix.drop(['Ticket'],axis=1)

# Add in missing values to complete the set
# Average out the age and then use this for missing age data
average_age=mix['Age'].mean()
mix['Age']=mix['Age'].fillna(average_age)

# Embarked (use most common port)
# Port of departure is wher ethe passanger left from# not sure how 
# this will impact the outcome but the cabin order or class could be 
# affected by the port of departure

port_of_departure = mode(mix['Embarked'].dropna())[0][0]
mix['Embarked'] = mix['Embarked'].fillna(port_of_departure)

# Fare need to average this for each class
# This gives us a better value and an equal playig field
# This allows us to put a ticket price in for the missiing values
average_ticket_price=mix.pivot_table('Fare',index='Pclass',aggfunc='mean')
mix['Fare'] = mix[['Fare', 'Pclass']].apply(lambda x:
                            average_ticket_price[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'],axis=1)
       
# Cabin we will use 0 for missing data and add this later 
mix['Cabin']=mix['Cabin'].fillna('0')
mix['Cabin']=mix['Cabin'].apply(lambda x: x[0])

      
# set the gender data to strings where 0 - male and 1 - female
mix['Sex']=mix['Sex'].map({'female':1,'male':0}).astype(int)

# Embarked This is interesting since in the training data
# Embarking from S had a 66% mortality rate of the total that left from S
# Embarking from Q had a 60% mortality rate of the total that left from Q 
# Embarking from C had a 45% mortality rate of the total that left from C 
# this tells us that leaving from S was would limit your chances based on the data 
# so here we will remove the Embark from S
mix = pd.concat([mix, pd.get_dummies(mix['Embarked'],  prefix='Embarked')], axis=1)
mix=mix.drop(['Embarked_S','Embarked'],axis=1)

# Cabin C cabin and down had a high mortality in the training data
# There is a fair amount of missing data here only a 1/4 is available
mix = pd.concat([mix, pd.get_dummies(mix['Cabin'], prefix='Cabin')], axis=1)
mix=mix.drop(['Cabin_0','Cabin'],axis=1)
           
# Name Grouping we will drop Mr         
mix['Title']=mix['Name'].apply(lambda x: re.split('[,.]',x)[1])
mix['Title']=mix['Title'].apply(lambda x: re.sub(' ','',x,1))

# Fix the name by grouping ms with Miss 
# Lady with Mlle, Mme, the countess, Dona  all the formal names
# Sir with Col Major, Capt, Jonkheer, Don, Dr  all the forma names 
# what is interesting here is that in the training set 6 out of 6 Reverends died
# I am guessing they stayed on the boat to comfort the remaining passanges.
# all the rest are set including the reverends to Mr.
mix['Title']=mix['Title'].apply(lambda x: 'Miss' if x=='Ms' else x)
mix['Title']=mix['Title'].apply(lambda x: 'Lady' if x in ('Mlle','Mme','the Countess','Dona')
                                            else x)
mix['Title']=mix['Title'].apply(lambda x: 'Sir' if x in ('Col','Major','Capt','Jonkheer','Don', 'Dr')
                                            else x)
mix = pd.concat([mix, pd.get_dummies(mix['Title'], prefix='Title')], axis=1)
mix=mix.drop(['Title_Mr','Title','Name'],axis=1)

# get the end timer and then calculate the time for data manipulation
# print this value
end = timer()
elapsed = end - start
print ('Time Data Manipulation: ', elapsed) 

# start the timer for measurement
start = timer()

# Split the combined data frame back into training and testing
# Now er have a much better set of complete data 
# A lot of processing is performed to fill in blanks 
training_dataframe=mix.head(891)
testing_dataframe=mix.tail(418)
testing_dataframe=testing_dataframe.drop(['Survived'],axis=1)

# Scale data for non-tree classifiers
mixnorm=mix
mixnorm.loc[:,['Pclass','Age','SibSp','Parch','Fare']]= (mixnorm - mixnorm.mean()) / (mixnorm.max() - mixnorm.min())
training_dataframenorm=mixnorm.head(891)
testing_dataframenorm=mix.tail(418)
testing_dataframenorm=testing_dataframenorm.drop(['Survived'],axis=1)

# Train the Classifiers on Training Data
# Convert data frames into numpy arrays
# Creating normals fro this set
train=training_dataframe.values
test=testing_dataframe.values
trainnorm=training_dataframenorm.values
testnorm=testing_dataframenorm.values

# get the end timer and then calculate the time for training
# print this value
end = timer()
elapsed = end - start
print ('Time Training: ', elapsed) 

# start the timer for measurement
start = timer()


# generate a random number for the forrest 
# the random number generator is the RandomState instance used by np.random
# So, basically, a sub-optimal greedy algorithm is repeated a number 
# of times using random selections of features and samples 
# (a similar technique used in random forests). 
# The random_state parameter allows controlling these random choices.
random.seed(400)

estimators = [10, 50, 100, 500, 1000, 5000, 10000]
for estimator in estimators:

    # Random Forest
    # Train on Training Set
    print ('')
    print ('Training Random Forest . . .')    
    
    # Using 100 estimator should be enough to provide speed and accuracy 
    # The higher the number the slower processing 
    # To low of a number provides less accuracy 
    # A 10 is 1% differnt that 100 but 1000 is the same as 100 but slower 
    random_forest_model = RandomForestClassifier(n_estimators = estimator)
    random_forest_model = random_forest_model.fit(train[0:,2:],train[0:,0])
    
    # Predict the Test Set
    print('')
    print('Predicting using Random Forest . . .')
    random_forest_output = random_forest_model.predict(test[:,1:])
    random_forest_output_train=random_forest_model.predict(train[:,2:]) 
         
    # Accuracy value 
    print ('')
    random_forest_predicted_value = metrics.accuracy_score(train[:,0].astype(int), random_forest_output_train)
    print ('Accuracy value: ', random_forest_predicted_value)
    
    # Feature Importance
    importances=random_forest_model.feature_importances_
    std=np.std([tree.feature_importances_ for tree in random_forest_model.estimators_],axis=0)
    importances_table=zip(training_dataframe.columns[2:], importances, std)
    sorted(importances_table, key=lambda x: x[1], reverse=True)
    print ("Feature : Importances : Standard Deviation")
    print (importances_table)
    
    # Write results to a csv file
    rfresult = np.c_[test[:,0].astype(int), random_forest_output.astype(int)]
    dfrfresult = pd.DataFrame(rfresult[:,0:2], columns=['PassengerId', 'Survived'])
    dfrfresult.to_csv('random_forest.csv', index=False) #change pathname
    
    end = timer()
    elapsed = end - start
    print ("Predicting Random Forest with Estimator %s and Time %s: " %(estimator, elapsed)) 
    
      # Accuracy value 
    print ('')
    random_forest_predicted_value = metrics.accuracy_score(train[:,0].astype(int), random_forest_output_train)
    print ('Accuracy value: ', random_forest_predicted_value)

# Feature Importance
importances=random_forest_model.feature_importances_
std=np.std([tree.feature_importances_ for tree in random_forest_model.estimators_],axis=0)
importances_table=zip(training_dataframe.columns[2:], importances, std)
sorted(importances_table, key=lambda x: x[1], reverse=True)
print ("Feature : Importances : Standard Deviation")
print (importances_table)

# Write results to a csv file
rfresult = np.c_[test[:,0].astype(int), random_forest_output.astype(int)]
dfrfresult = pd.DataFrame(rfresult[:,0:2], columns=['PassengerId', 'Survived'])
dfrfresult.to_csv('random_forest.csv', index=False) 
