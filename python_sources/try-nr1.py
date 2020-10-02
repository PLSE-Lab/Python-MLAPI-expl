import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train = pd.read_csv("../input/train.csv")

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

print(train[['Sex', 'Pclass','Survived']].groupby(['Sex','Pclass'], as_index=False).mean())

print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
print(train[['Parch']].groupby(['Parch'], as_index=False))