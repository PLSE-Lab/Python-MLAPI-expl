import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#Read csv to DataFrame
train_df = pd.read_csv("../input/train.csv")


#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train_df.head())

print("\n\nSummary statistics of training data")
print(train_df.describe())
