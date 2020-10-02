import numpy as np
import pandas as pd
import pylab as pl

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#--------------------------
#--- using data frames
#--------------------------

#Print to standard output, and see the results in the "log" section below after running your script
'''
print("\n\nTop of the training data:")
print(train.head(3))

print("\n\nSummary statistics of training data")
print(train.describe())

print(train[train['Age'] > 60][['Age','Sex','Pclass','Survived']])
print(train[train['Age'].isnull()][['Age','Sex','Pclass','Survived']])
'''

#Take a count of the males in each class
'''
for i in range(1,4):
    print(i,len(train[(train['Sex']=='male') & (train['Pclass']==i)]))
'''

# make histogram

train['Age'].hist()
pl.show()

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)