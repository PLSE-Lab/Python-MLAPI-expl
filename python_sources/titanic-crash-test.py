import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train['Gender'] = train['Sex'].map({'female':0, 'male':1}).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                            (train['Pclass'] == j+1)]['Age'].dropna().median()
print(median_ages)

train['AgeFill'] = train['Age']

for i in range(0,2):
    for j in range(0,3):
         train.loc[(train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+i),\
         'AgeFill'] = median_ages[i,j]

print(train[train['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10))

#print(train.loc[train.Age.isull()][['Gender','Pclass','Age']])