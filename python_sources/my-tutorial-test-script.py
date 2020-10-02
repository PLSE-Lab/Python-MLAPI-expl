import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

print(train.dtypes)
print(train.info())

print(train['Age'][0:10])
print(train[['Sex', 'Pclass', 'Age']])

print(train[train["Age"].isnull()][['Sex', 'Pclass', 'Age']])

import pylab as P
train['Age'].hist()
P.show()