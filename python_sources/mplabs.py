import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())
print(train['Cabin'].isnull().count())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

train['Age'] = train['Age'].fillna(value=train['Age'].mean())

for a in train.columns:
    per = ((train[a].isnull().sum()/train[a].count())*100)
    count = train[a].isnull().sum()
    print(a,count,per)