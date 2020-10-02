import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#clf = RandomForestClassifier(n_estimators=10)
#clf.fit(train)
print(train.PassengerId)

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)