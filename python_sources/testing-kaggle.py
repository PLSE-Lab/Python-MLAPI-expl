import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train['Gender'] = train['Sex'].map( {'female':0,'male':1} ).astype(int)
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.loc[ (train.Embarked.isnull()), 'Embarked' ] = train.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(train.Embarked)))
Port_dict = {name: i for i,name in Ports}
train.Embarked = train.Embarked.map(lambda x: Port_dict[x]).astype(int)

print(train.head(10))

median_age = train.Age.dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age' ] = median_age

print(train.head(10))


#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

#Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)