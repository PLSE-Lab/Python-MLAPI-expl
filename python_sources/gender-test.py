

import pandas as pd
import numpy as np

# Read in data
train = pd.read_csv('../input/train.csv')

total = float(train.shape[0])
p_survive = train[train['Survived'] == 1].shape[0] / total

# Split on gender and calculate probs
female = train[train['Sex'] == 'female']
p_female = female.shape[0] / total
female_survived = female[female['Survived'] == 1]
p_female_survived = female_survived.shape[0] / total

male = train[train['Sex'] == 'male']
p_male = male.shape[0] / total
male_survived = male[male['Survived'] == 1]
p_male_survived = male_survived.shape[0] / total

# P(Survive|Woman) = P(Woman|Survive)*P(Survive)/P(Woman)
survived = (female_survived.shape[0] + male_survived.shape[0]) / total
print("P Woman", p_female)
print("P Survived", survived)
print("P Woman given Survived", p_female_survived)

print("Female probability of survival: {}".format(p_female_survived))
print("Male probability of survival: {}".format(p_male_survived))

test = pd.read_csv('../input/test.csv')

test['Survived'] = 0

test[test['Sex'] == 'female']['Survived'] = 1

output = test[['PassengerId', 'Survived']]

output.to_csv('gendermodel.csv', index=False)