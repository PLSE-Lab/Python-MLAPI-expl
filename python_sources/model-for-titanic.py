import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train['AgeFill'] = train['Age']

median_ages = np.zeros((2,3))

### make a median_ages table
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                              (train['Pclass'] == j+1)]['Age'].dropna().median()
                              
### reference the median ages table 
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

train['AgeIsNull'] = pd.isnull(train.Age).astype(int)

train['Age*Class'] = train.AgeFill * train.Pclass
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 

### delete rows with any NA values
train = train.dropna()

print(train.head())

### building the model
#### for a random forest everything needs to be in a factor.. also we cannot have any missing data
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train[0::,1::],train[0::,0])

output = forest.predict(test)



#Any files you save will be available in the output tab below
##train.to_csv('copy_of_the_training_data.csv', index=False)



