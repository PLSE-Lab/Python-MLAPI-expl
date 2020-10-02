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

fare = train["Fare"]
plt.subplot(3, 1, 1);
#fare[train["Survived"] == 1].hist(bins=40, color="green", align="mid", rwidth=0.3)
#fare[train["Survived"] == 0].hist(bins=40, color="red", align="right", rwidth=0.3)
plt.hist((fare[train["Survived"] == 1].values, fare[train["Survived"] == 0].values), bins=40)

cl = train["Pclass"]
plt.subplot(3, 1, 2);
#cl[train["Survived"] == 1].hist(bins=3, range=(1,4), color="green", align="mid", rwidth=1.0)
#cl[train["Survived"] == 0].hist(bins=3, range=(1,4), color="red", align="right", rwidth=0.5)
plt.hist((cl[train["Survived"] == 1].values, cl[train["Survived"] == 0].values), bins=3, range=(1, 4), align="mid")

plt.subplot(3, 1, 3);
sex_survived = train[train["Survived"] == 1].groupby("Sex").size()
sex_loss = train[train["Survived"] == 1].groupby("Sex").size()
plt.bar((0, 2), (sex_survived["female"], sex_survived["male"]), color="blue")
plt.bar((1, 3), (sex_loss["female"], sex_loss["male"]), color="green")

plt.show()
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
