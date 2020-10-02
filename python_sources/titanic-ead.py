import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

#Print number of NAs in each column
print(train.isnull().sum())

#Variables: Passenger Class, Name, sex, age, parents/children, siblings/spouse, ticket number, fare, cabin, embarked

#Barplots
#1. Target Variable
train.Survived.value_counts().plot(kind='bar')
