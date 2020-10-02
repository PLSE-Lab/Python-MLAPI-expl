#!/usr/bin/env python
# coding: utf-8

# #Boarding A Sinking Ship
# #### A Deep Dive Into The Unknown Waters of Machine Learning
# 
# ## Introduction
# 
# Having done a data science online course for a few months I have decided to try and put what I have learned to the test rather than just completing assignment after assignment.
# 
# Having looked around Kaggle and with the advise of a couple of friends I have decided to have a crack at the Titanic Dataset.
# 
# ## Aim
# 
# My main aim will be to create a machine learning script that can predict whether a passenger survived dependent on a certain array of variables. I will do this by employing a random forest approach (as recommended in the competition!)
# 
# ## Step 1: Importing Libraries
# 
# I will first import all of the necessary libraries I'll need

# In[ ]:


#import the below mathematical libraries
import pandas as pd
import numpy as np
import csv as csv

#import libraries for plotting data
import matplotlib.pyplot as plt
import seaborn as sb

#import machine learning libraries
from sklearn.ensemble import RandomForestClassifier 


# ## Step 2: Import & Check the Dataset
# 
# I will now import the training data (train.csv) and use pd.describe() to get a summary of all of the fields

# In[ ]:


#Read in the train.csv file
train_data = pd.read_csv("../input/train.csv")

#Print the columns and their corresponding types
print(train_data.info())

#Print a summary of all the numerical fields
print(train_data.describe())


# As can be seen a runtime error is thrown due to the fact that the age field has unknown values. From this I can see two options:
# 
# 1. Delete all the rows with NaN values. Though this could potentially skew the data.
# 
# 2. We can estimate the ages for rows with NaN values using the median values of passengers on board.
# 
# To decide this I will look at the distribution of ages for this patient.

# In[ ]:


# We have to temporarily drop the rows with 'NA' values
# because the Seaborn plotting function does not know
# what to do with them

sb.pairplot(train_data[['Sex','Pclass', 'Age', 'Survived']].dropna(), hue='Sex')


# As can be seen the age seems to be positively skewed, meaning it will probably be best for us to replace the ages rather than remove the ones with missing ages. 
# 
# What is also interesting to notice is the gender of those who died compared to those who survived. Those who died were primarily male and those who survived were primarily female.
# 
# So we now have a course of action, I plan to create an estimated age for all the null age values, I will also give the sex field a numeric value so that these fields can be utilised by the random forest process.
# 
# ## Step 3: Data Munging
# 
# ### Age_All & Age_Estimated Fields
# 
# I will first create the estimated age field. I am going to do this by calculating the median age for each passenger class and then use these medians to replace any age that we don't have.

# In[ ]:


#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
pclass_values = train_data['Pclass'].unique()
median_age_pclass = {}

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_age_pclass[i] = train_data[train_data['Pclass'] == i]['Age'].median()

#print dictionary to check results are as expected
print(median_age_pclass)

#We now have the median value for each task!


# We now have the medians. So I am now going to create an additional field called Age_All which will include the estimated along with the actual ages we have. I'll also include a flag called Age_Estimated that denotes whether the value in Age_All is estimated or not. 

# In[ ]:


#Create the new age_all field which is initially a direct copy of Age
train_data['Age_All'] = train_data['Age']

#Create the Age_Estimated Flag
train_data['Age_Estimated'] = pd.isnull(train_data['Age']).astype(int)

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    train_data.loc[(train_data['Age_Estimated'] == 1) & (train_data['Pclass'] == i),'Age_All'] = median_age_pclass[i]

#Print the top 20 rows of a few select columns to spot check if they correspond accordingly
print(train_data[['Sex','Pclass','Age','Age_All','Age_Estimated','Survived']].head(20))


# In[ ]:





# In[ ]:


#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
median_fare_pclass = {}

#Create subset of data where Fare > 0
td_fare = train_data.loc[train_data['Fare'] >0,('Pclass','Fare')]

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_fare_pclass[i] = td_fare[td_fare['Pclass'] == i]['Fare'].median()
    
#Create the new age_all field which is initially a direct copy of Age
train_data['Fare_All'] = train_data['Fare']

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    train_data.loc[((train_data['Fare_All'] == 0) | (train_data['Fare_All'].isnull())) & (train_data['Pclass'] == i),'Fare_All'] = median_fare_pclass[i]    

#Print the all rows where Fare = 0
print(train_data.loc[(train_data['Fare'] == 0),('Sex','Pclass','Fare','Fare_All','Survived')])


# As can be seen just quickly overlooking the table we have successfully created both the Age_Estimated abd Age_All fields. 
# 
# This means we can now go on to creating a numeric field that represents the string! Once we have done this we can create our cleaned dataset and get on with some Machine Learning!!!
# 
# ### Numerical Field with Sex Data
# 
# Here I am going to map values from the Sex field, give them a numerical value and then put it into the newly created Gender Field.
# 
# The mapping will be as follows, 'female' = 1 and 'male' = 2

# In[ ]:


#Create a mapping dictionary
Sex_Map = {'female': 1,'male': 2}
Embarked_Map = {'Q':1,'C':2,'S':3,'X':0}

#Set all NaN values in Embarked Field = X
train_data.loc[(train_data['Embarked'].isnull()),'Embarked'] = 'X'

#Use mapping dictionary to map values from 'Sex' field to the new field 'Gender'
train_data['Gender'] = train_data['Sex'].map(Sex_Map).astype(int)

#Use mapping dictionary to map values from 'Embarked' field to the new field 'Embarked_From'
train_data['Embarked_From'] = train_data['Embarked'].map(Embarked_Map).astype(int)

#Print the top 20 rows of a few select columns to spot check if they correspond accordingly
print(train_data[['Sex','Gender','Embarked','Embarked_From','Survived']].head(20))


# We now have a gender column! Now onto the final step.
# 
# ### Creating Cleaned Dataset
# 
# We now have all of the fields we will require for use in our Random Forest Algorithm. We now need to reduce the columns down to just the ones that are int or float and then transfer the dataset from the Dataframe we are currently using to a NumPy array.
# 
# I am firstly going to analyse the all the columns we have available

# In[ ]:


#Print the columns and their corresponding types
print(train_data.info())


# From this I think we will make the cleaned dataset as follows:
# 
# 1. Survived
# 2. Pclass
# 3. Age_All
# 4. Age_Estimated
# 5. Gender
# 6. SibSp
# 7. Parch
# 8. Fare_All
# 9. Embarked_From
# 
# I will now create this dataset.
# 

# In[ ]:


#Make the clean dataset and put it into a NumPy array
train = train_data[['Survived','Pclass','Age_All','Age_Estimated','Gender','SibSp','Parch','Fare_All','Embarked_From']].values

#Print top 5 rows of NumPy array to double check the format is correct
print(train[:][0:5])


# Thats all good! So now that we have the cleaned training dataset we are now going it to train the random forest algorithm. We will then run it on the test data to see how accurate our model is.
# 
# ## Step 4: Machine Learning
# 
# ### Cleaning Test Data
# 
# So the first thing we are going to do is clean the test data in the same fashion that we cleaned the training data. This can be seen below

# In[ ]:


#Read in the test.csv file
test_data = pd.read_csv("../input/test.csv")

#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
median_test_pclass = {}

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_test_pclass[i] = test_data[test_data['Pclass'] == i]['Age'].median()

#Create the new age_all field which is initially a direct copy of Age
test_data['Age_All'] = test_data['Age']

#Create the Age_Estimated Flag
test_data['Age_Estimated'] = pd.isnull(test_data['Age']).astype(int)

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    test_data.loc[(test_data['Age_Estimated'] == 1) & (test_data['Pclass'] == i),'Age_All'] = median_age_pclass[i]
    
#Create the new age_all field which is initially a direct copy of Age
test_data['Fare_All'] = test_data['Fare']

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    test_data.loc[((test_data['Fare_All'] == 0) | (test_data['Fare_All'].isnull())) & (test_data['Pclass'] == i),'Fare_All'] = median_fare_pclass[i]    
    
#Set all NaN values in Embarked Field = X
test_data.loc[(test_data['Embarked'].isnull()),'Embarked'] = 'X'

#Use mapping dictionary to map values from 'Sex' field to the new field 'Gender'
test_data['Gender'] = test_data['Sex'].map(Sex_Map).astype(int)

#Use mapping dictionary to map values from 'Embarked' field to the new field 'Embarked_From'
test_data['Embarked_From'] = test_data['Embarked'].map(Embarked_Map).astype(int)

#Make the clean dataset and put it into a NumPy array
test = test_data[['Pclass','Age_All','Age_Estimated','Gender','SibSp','Parch','Fare_All','Embarked_From']].values

#Make a table with the passenger ids for use in the output file
PasID = test_data['PassengerId']

#Print top 5 rows of NumPy array to double check the format is correct
print(test[:][0:6])


# In[ ]:


print(test_data.loc[(test_data['Fare_All'].isnull()),'Fare_All'])


# We now have the test file in a NumPy Array. Do you know what this means?! We can finally get to the machine learning!!!
# 
# ### Run Random Forrest Algorithm
# 
# I will now train the algorithm and then apply it to the test data and get it to predict the survived value.

# In[ ]:


print('Training!')

#Create the random forest object which will include all the parameters
#for the fit
forest = RandomForestClassifier(n_estimators = 1000)

#Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train[0::,1::],train[0::,0])

print('Test!')

#Take the same decision trees and run it on the test data
output = forest.predict(test)

#Print top 5 rows
print(output[:][0:5])


# And thats all been done successfully! Now we need to output a CSV file so that we can submit the answers to Kaggle and test the accuracy of the data.
# 
# ### Export CSV File
# 
# The rules state that we need a csv file with 418 rows (the number of rows in the test data) with two columns, PassengerID and Survived. 

# In[ ]:


#Create Dataframe with correct structure and using the output and ids arrays
predictions_file = pd.DataFrame({
        "PassengerId": PasID
       ,"Survived"   : output
    })

#Write DataFrame to CSV
predictions_file.to_csv('submission.csv', index=False)

#Print first 5 rows of file and close
print(predictions_file.head())


# We have now exported the CSV. Now to submit I will summarise the result I get below
# 
# ## Results:
# ### Score: 0.29
# ### To-Do: 
# 
# 1. Could possibly add fares back in once I remove the nulls or add predicted results. 
# 2. Add a numerical input for departure origin
# 3. Possibly use a better machine learning algorithm
