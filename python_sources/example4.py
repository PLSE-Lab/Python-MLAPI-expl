import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')
#print(train["Age"].describe())

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
print(train[train["Age"] < 18].describe())
train["Child"][train["Age"] < 18] = 1
# Compare with previous print to show if the new column works
print(train[train["Child"] == 1].describe())
print("\n\nNew colume Child:")
print(train[train["Child"] == 1])



# Print normalized Survival Rates for passengers under 18
print("\n\nNormalized Survival Rates for passengers under 18:")
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print("\n\nNormalized Survival Rates for passengers older than 18:")
print(train["Survived"][train["Child"] != 1].value_counts(normalize = True))
