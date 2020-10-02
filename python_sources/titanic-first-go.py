import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

del train['Name']
del test['Name']

del train['Ticket']
del test['Ticket']
'''
#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
#print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
print('-'*40)
print(train.info())
'''
print(('-'*40+'\n')*3)

print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
train.plot.scatter(x='Age', y='Survived')

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)