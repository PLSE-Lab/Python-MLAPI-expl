import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

sns.set_style('whitegrid')

#Read in CSV
train = pd.read_csv("../input/train.csv",dtype={"Age": np.float64,"Fare":np.float64})
test = pd.read_csv("../input/test.csv",dtype={"Age": np.float64,"Fare":np.float64})

#Remove Unecessary Columns
train=train.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)
test = test.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)

# Create arrays
train_data = np.array(train)
test_data = np.array(test)

#Clean Data
train["Age"]=train["Age"].fillna(0)
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Sex"][train["Sex"] == "female"] = 0
train["Sex"][train["Sex"] != "female"] = 1

#Create Masks
Women_Survived = train_data[0::,2]== 0
Men_Survived = train_data[0::,2]== 1
Rich_Survived = train_data[0::,1] == 1
Average_Survived = train_data[0::,1] == 2
Poor_Survived = train_data[0::,1] == 3
Young_Survived = train_data[0::,3] < 18

#Create Variables
number_passengers = np.size(train_data[0::,0].astype(np.float))
number_survivors = np.sum(train_data[0::,0].astype(np.float))
women_survivors_onboard = train_data[Women_Survived,0].astype(np.float)
men_survivors_onboard = train_data[Men_Survived,0].astype(np.float)

#Plot Data

#Create Statistics
proportion_survivors =train["Survived"].value_counts(normalize=True)
proportion_women_survivors = train["Survived"][Women_Survived].value_counts(normalize=True)
proportion_men_survivors = train["Survived"][Men_Survived].value_counts(normalize=True)

#Passenger Class

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

print(proportion_men_survivors)

print(proportion_women_survivors)
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
'''

import numpy as np
import pandas as pd


#Read in CSV
train = pd.read_csv("../input/train.csv",dtype={"Age": np.float64,"Fare":np.float64})
test = pd.read_csv("../input/test.csv",dtype={"Age": np.float64,"Fare":np.float64})

data = np.array(train)								# Then convert from a list to an array.

# Now I have an array of 12 columns and 891 rows
# I can access any element I want, so the entire first column would
# be data[0::,0].astype(np.float) -- This means all of the rows (from start to end), in column 0
# I have to add the .astype() command, because
# when appending the rows, python thought it was a string - so needed to convert

print(data)

# Set some variables
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers 

# I can now find the stats of all the women on board,
# by making an array that lists True/False whether each row is female
women_only_stats = data[0::,4] == "female" 	# This finds where all the women are
men_only_stats = data[0::,4] != "female" 	# This finds where all the men are (note != means 'not equal')

# I can now filter the whole data, to find statistics for just women, by just placing
# women_only_stats as a "mask" on my full data -- Use it in place of the '0::' part of the array index. 
# You can test it by placing it there, and requesting column index [4], and the output should all read 'female'
women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

# and derive some statistics about them
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print ('Proportion of women who survived is %s' % proportion_women_survived)
print ('Proportion of men who survived is %s' % proportion_men_survived)

'''

