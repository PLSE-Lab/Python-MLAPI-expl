import numpy as np
# import csv as csv

# #Print you can execute arbitrary python code

# csv_file_object = csv.reader(open('../input/train.csv', 'r'))
# csv_file_object.__next__()

# data = []
# for row in csv_file_object:
#     data.append(row)
# data = np.array(data)

# print(data)

# print(data[0:15,5])

import pandas as pd

import pylab as P


df = pd.read_csv('../input/train.csv', header=0)

print(df.head(3)) 
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.Age.median())
print(type(df['Cabin']))
print(df[df['Age']>60])

df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
print(df['Gender'])


# train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
# test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# #Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)