# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Part 1 - Preprocessing the data

# Importing the test dataset
dataset_test = pd.read_csv('../input/test.csv')

# Importing the training dataset
dataset_train = pd.read_csv('../input/train.csv')

# Joining both datasets
df1 = pd.DataFrame(dataset_train)
df2 = pd.DataFrame(dataset_test)
dataset_joined = [df1, df2]
dataset_joined = pd.concat(dataset_joined)


# Cleaning the data

# removing everything from the names before the comma and after the dot, so we have only titles
dataset_joined['Name'] = dataset_joined['Name'].str.split(',').str[1]
dataset_joined['Name'] = dataset_joined['Name'].str.split('.').str[0]        

# droping irrelevant data
dataset_joined.drop(['Ticket', 'Cabin' ], axis = 1, inplace = True)

# Filling nan values Embarked with 'S', the most relevant data
dataset_joined['Embarked'].fillna('S', inplace=True)

dataset_joined.info()



# Enconding Categorical Data

# Categorizing the 'Name' data by treatment hierarquy and gender segregation
names = dataset_joined['Name'].copy()
for item in names:
    if (item == ' Mr'):
        names.replace(item, 1, inplace = True)
    elif (item == ' Miss' or item == ' Mrs'):
        names.replace(item, 0, inplace = True)
    elif (item == ' Capt' or item == ' Col' or item == ' Don' or item == ' Dona' or item == ' Dr' or item == ' Jonkheer' or item == ' Lady' or item == ' Major' or item == ' Master' or item == ' Mile' or item == ' Mlle' or item == ' Mme' or item == ' Ms' or item == ' Rev' or item == ' Sir' or item == ' the Countess'):
        names.replace(item, 2, inplace = True)
    
dataset_joined['Name'] = names




from sklearn.preprocessing import LabelEncoder

labelencoder_sex = LabelEncoder()
dataset_joined['Sex'] = labelencoder_sex.fit_transform(dataset_joined['Sex'])

labelencoder_embarked = LabelEncoder()
dataset_joined['Embarked'] = labelencoder_embarked.fit_transform(dataset_joined['Embarked'])



# Estimating and updating Age and Fare for the null values on dataset


# Getting average Age and Fare
# There are two genders and three passenger classes in this dataset. 
# So we create a 2 by 3 matrix to store the median values.
 
# Create a 2 by 3 matrix of zeroes
median_ages = np.zeros((2,3))
median_fares = np.zeros((2,3))
 
# For each cell in the 2 by 3 matrix
for i in range(0,2):
    for j in range(0,3):
 
    	# Set the value of the cell to be the median of all `Age` values
    	# matching the criterion 'Corresponding gender and Pclass',
    	# leaving out all NaN values
        median_ages[i,j] = dataset_joined[ (dataset_joined['Sex'] == i) & \
                               (dataset_joined['Pclass'] == j+1)]['Age'].dropna().median()
        median_fares[i,j] = dataset_joined[ (dataset_joined['Sex'] == i) & \
                               (dataset_joined['Pclass'] == j+1)]['Fare'].dropna().median()

# Create new columns AgeFill and FareFill to put values into. 
# This retains the state of the original data.
dataset_joined['AgeFill'] = dataset_joined['Age']
dataset_joined[ dataset_joined['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']]
dataset_joined['FareFill'] = dataset_joined['Fare']
dataset_joined[ dataset_joined['Fare'].isnull()][['Fare', 'AgeFill', 'Sex', 'Pclass']]

# Put our estimates into NaN rows of new columns AgeFill and FareFill.
# df.loc is a purely label-location based indexer for selection by label.
 
for i in range(0, 2):
    for j in range(0, 3):
 
    	# Locate all cells in dataframe where `Sex` == i, `Pclass` == j+1
    	# and `Age` == null and 'Fare' == null. 
    	# Replace them with the corresponding estimate from the matrix.
        dataset_joined.loc[ (dataset_joined.Age.isnull()) & (dataset_joined.Sex == i) & (dataset_joined.Pclass == j+1),\
                 'AgeFill'] = median_ages[i,j]	
        dataset_joined.loc[ (dataset_joined.Fare.isnull()) & (dataset_joined.Sex == i) & (dataset_joined.Pclass == j+1),\
                 'FareFill'] = median_fares[i,j]	
        

# Create a feature that records whether the Age was originally missing
dataset_joined['AgeIsNull'] = pd.isnull(dataset_joined['Age']).astype(int)
dataset_joined['FareIsNull'] = pd.isnull(dataset_joined['Fare']).astype(int)
dataset_joined.head()


# Now we remove the null values from the test dataset and we clean the columns Age and Fare
dataset_joined.drop(['Age', 'Fare' ], axis = 1, inplace = True)

# Filling no Survived data with -1
dataset_joined['Survived'].fillna(-1, inplace=True)

dataset_joined.info()



# The 'Embarked', 'Name' and 'Pclass' columns has several types of data, with no value Hierarchy, so we need to
# create dummy variables and exclude 1 one of them to avoid the dummy variable trap

dataset_joined = pd.get_dummies(dataset_joined, columns=['Embarked', 'Name', 'Pclass'], drop_first=True)


# Splitting the dataset into Train and Test
dataset_train_revised = dataset_joined.iloc[:891, :]
dataset_test_revised = dataset_joined.iloc[891:, :]

# Splitting the dataset into the input and output
X_train = dataset_train_revised.iloc[:,[0,2,3,5,6,9,10,11,12,13,14]]
y_train = dataset_train_revised.iloc[:, [4]]
X_test = dataset_test_revised.iloc[:,[0,2,3,5,6,9,10,11,12,13,14]]

#getting the PassIndex for the Submission dataset
pass_index = dataset_joined.iloc[891:,1]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Part 2 - Making the predictions

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0, max_depth = 8, min_samples_split = 5, min_samples_leaf = 5)
classifier.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)



#Part 3 - Generating Submission File
only_final_values = pd.read_csv('../input/test.csv')
only_final_values['Survived'] = y_pred
only_final_values = only_final_values.iloc[: , [0,11]]
only_final_values['Survived'] = only_final_values['Survived'].astype(np.int64)
only_final_values.to_csv('titanic_submission_random_forest.csv', index = False)


