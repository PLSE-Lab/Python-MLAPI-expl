import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
titanic = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(titanic.head())

print("\n\nSummary statistics of training data")
print(titanic.describe())

# Find all the unique genders -- the column appears to contain only male and female.
print(titanic["Sex"].unique())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1



# replace na's with median
print(titanic.loc[titanic["Sex"]==1,"Age"].median())
print(titanic.loc[titanic["Sex"]==0,"Age"].median())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#titanic.loc[titanic["Sex"]==1,"Age"] = titanic.loc[titanic["Sex"]==1,"Age"].fillna[titanic.loc[titanic["Sex"]==1,"Age"].median())

import matplotlib.pyplot as plt
plt.boxplot(titanic["Age"])
plt.show()

#Any files you save will be available in the output tab below
titanic.to_csv('copy_of_the_training_data.csv', index=False)