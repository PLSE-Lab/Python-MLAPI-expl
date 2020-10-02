import numpy as np
import pandas as pd
import pylab as Plb

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


#TRAIN DATA SET

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())
#
#print("\n\nSummary statistics of training data")
#print(train.describe())

#print(train[train.Age > 60][ ['Age', 'Sex'] ])

#print(train[train.Age.isnull()][ ['Sex', 'Pclass', 'Age'] ])

#train['Gender'] = 4
#print(train.head())

#train['Gender'] = train['Sex'].map({'female': 0, 'male':1}).astype(int)
print(train['Embarked'])


#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)