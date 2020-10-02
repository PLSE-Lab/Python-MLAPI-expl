import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, header=0 )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, header = 0)

train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
print(train.dtypes)

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)