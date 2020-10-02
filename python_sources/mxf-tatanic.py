import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]

print(train_df.columns.values)

print (train_df.head())

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)