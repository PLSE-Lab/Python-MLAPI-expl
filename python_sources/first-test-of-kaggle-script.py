import numpy as np
import pandas as pd
import seaborn as sns
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
train.to_csv('copy_of_the_training_data.csv', index=False)

# Replacing missing ages with median
train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])
train["Fare"][np.isnan(train["Fare"])] = np.median(train["Fare"])
train["Pclass"][np.isnan(train["Pclass"])] = np.median(train["Pclass"])

plt.figure()

smalltrain = train[["Fare","Pclass", "Survived"]]
print("Describe the small trainning data")
print(smalltrain.describe())
sns.pairplot(smalltrain, hue="Survived")
plt.savefig("1_seaborn_pair_plot.png")