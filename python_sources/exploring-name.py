import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from collections import Counter

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#extract title
train['Title'] = train['Name'].str.extract(',\s(\w+)\.')
#print(train[['Survived','Title']].sort_values(['Title']))
train.loc[(train['Title'] != 'Mrs') & (train['Title'] != 'Mr') & (train['Title'] != 'Master') & (train['Title'] != 'Miss'),'Title'] = 'Rare'

train['Title_code'] = pd.Categorical(train['Title']).codes
print(train['Title_code'])

'''
#extract cabin
train['Cabin'] = train['Cabin'].str.extract('^(\w)')
train['Cabin_code'] = pd.Categorical(train['Cabin']).codes
print(train['Cabin_code'])
'''
