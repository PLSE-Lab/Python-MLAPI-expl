# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Example Data One
print('\n~~~~~~~~~~~Example 1 ~~~~~~~~~~~\n')        
raw_data = {'patient': [1, 1, 1, 2, 2],
                'obs': [1, 2, 3, 1, 2],
                'treatment': [0, 1, 0, 1, 0],
                'score': ['strong', 'weak', 'normal', 'weak', 'strong']}

df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])
print(df)

le = LabelEncoder()

def encodeCategorical(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

df = encodeCategorical(df)

print('\n=======Result after converted=======\n')
print(df)

#Example Data Two
print('\n~~~~~~~~~~~Example 2 ~~~~~~~~~~~\n')
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                'age': [42, 52, 36, 24, 73],
                'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}

df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
print(); print(df)

print();print(pd.get_dummies(df['city']))

processed_data = le.fit_transform(df['city'])
print(); print(processed_data)